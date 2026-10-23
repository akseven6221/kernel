# v2：TMEM → reg → shared → TMA → global

[v2.cu](v2.cu) 在原有 CuTe MMA 主循环基础上，采用
[cuda_kernel/v3.cu](../cuda_kernel/v3.cu) 的输出布局和 TMA 写回方式。
命名空间为 `v2`。计算仍为 `C[M,N] = A[M,K] × B[N,K]ᵀ`，
BF16 输入/输出，FP32 累加，CTA 为 128 线程，tile 为 `64×256×64`。

## 写回流程

1. `Copy_Atom<SM100_TMEM_LOAD_16dp256b32x, float>` 从 TMEM 读结果，
   每个线程得到 `fragment[128]`，等待 TMEM load 完成。
2. 将相邻两个 FP32 转为一对 BF16，打包为一个 `uint32_t`。
   `SM90_U32x4_STSM_N::copy` 使用四个打包值执行 `stmatrix.m8n8.x4`。
3. 输出 shared 区由四个独立的 `64×64` BF16 tile 组成，地址使用 128B swizzle：
   `offset ^ ((offset >> 3) & 0x70)`。fragment 顺序和 lane 地址与 CUDA v3 一致。
4. 所有线程执行 `tma_store_fence()`，再 `__syncthreads()`，发布 shared 写入，
   并确保整个 CTA 完成 TMEM 读取和输出暂存。
5. 线程 0～3 各调用一次 `SM90_TMA_STORE_2D::copy`，从 shared 搬一个 `64×64`
   tile 到 C，对应坐标 `(blockIdx.x*256 + threadIdx.x*64, blockIdx.y*64)`。
   各线程提交自己的 bulk group 并等待 store 完成，随后 warp 0 释放 TMEM。

`cute::tma_store_wait<0>()` 在本版本 CUTLASS 中生成 `wait_group.read`，
只等待 shared 源读取完成。为与 CUDA v3 的完整 store 等待一致，本实现保留
`asm volatile("cp.async.bulk.wait_group 0;" ::: "memory")`。
另两处 `tcgen05.fence::after_thread_sync` 也沿用原实现。

相关封装见官方
[stmatrix 接口](https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/arch/copy_sm90.hpp)、
[TMA store、fence 和 wait 接口](https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/arch/copy_sm90_tma.hpp)。

## 调用接口

`gemm` 和 `launch` 现在接收输出 tensor map `c_map`。
用 `encode_output` 编码 C，连续维为 N，box 为 `{64,64}`，采用 128B swizzle。
下面的 `check` 为 benchmark 中的错误检查函数；A/B map 按原接口编码。

```cpp
alignas(64) CUtensorMap c_map;
check(v2::encode_output(&c_map, output, rows, columns));
check(cudaFuncSetAttribute(v2::gemm,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          v2::SHARED_BYTES));
v2::launch(a_map, b_map, c_map, rows, columns, reduction, stream);
check(cudaGetLastError());
```

动态 shared 布局为 `[A 8192B][B 32768B][C 32768B][barrier/地址 24B]`，
共 **73752B**。启动前必须设置上面的动态 shared 内存上限。
支持正的 M/N/K，且 `M%64=N%256=K%64=0`。

## B200 验证与性能

2026-10-09 北京时间 16:08，Modal NVIDIA B200（148 SM），CUDA 12.8.1，
nvcc 12.8.93，cuBLAS 12.8.4，驱动 580.95.05，CUTLASS 4.2.1，
commit `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`。
编译为 `-std=c++17 -O3 -lineinfo -arch=sm_100a`，保留默认断言。

- `64×256×64`、`128×512×192`、`192×768×576`、`4096×4096×64` 和三次 `4096³`
  均通过 identity 精确对照、随机输入全量对照 cuBLAS，以及 32 个 CPU FP64 采样检查。
- 随机输入输出与 cuBLAS 一致，`max_abs_error=0`、`relative_l2=0`。
- `192×512×576` 的 memcheck/synccheck 均为 0 errors；racecheck 为
  0 hazards、0 errors、0 warnings。额外 `4096×4096×576` racecheck 也全部为零。
- CuTe v2 与 PTX v3 均使用 137 registers，均为 0 stack、0 spill。
- [生成的 PTX](write_v2.sm100a.ptx) 包含 TMEM load、16 条展开的 stmatrix、TMA store、
  proxy fence、bulk commit 和完整 bulk wait；本地源码 SHA256 与远程运行一致。

`M=N=K=4096`，三次 run 中位数的中位数：

| 实现 | 耗时 | 吞吐量 |
| --- | ---: | ---: |
| CuTe v2 | 0.496229 ms | 276.97 TFLOPS |
| 原版 PTX v3 | 0.488581 ms | 281.30 TFLOPS |

本次 CuTe v2 耗时比 PTX v3 高 1.57%。三次耗时分别为
0.492322/0.496257/0.496229 ms；PTX v3 为 0.488006/0.501992/0.488581 ms。
测量存在波动，本次结果不足以判断该差异是否稳定。

复用原 `benchmark_v3.cu`，仅替换 include、命名空间别名和显示标签。
同一 B200 上按 AB/BA/AB 顺序比较；每次直接预热 100 次，交替预热各 50 次
CUDA Graph replay，每个 replay 执行 100 次 GEMM，CUDA events 计时，
custom/cuBLAS 交替 7 轮取中位数，最后取三次 run 中位数的中位数。

复现：

```bash
.venv/bin/modal run kernel/write_by_myself2/run_v2_modal.py --sanitize
```

[完整结果及日志](results_v2_b200_4096x4096x4096.json)；
[Modal 运行记录](https://modal.com/apps/akseven6221/main/ap-nUot0XYUBVGXPWEss3HGUD)。
