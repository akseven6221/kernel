# v1 修正与 B200 benchmark

v2 的 TMEM → reg → shared → TMA → global 写回实现见 [v2 说明](README_v2.md)。

[v1.cu](v1.cu) 计算 `C = A × Bᵀ`：A 按 `[M,K]`、B 按 `[N,K]` 行主序存储，
输入和输出为 BF16，累加为 FP32。每个 CTA 使用 128 个线程计算 `64×256` 输出块，
每轮处理 `K=64`，使用单 shared 缓冲与 512 列 TMEM。

## 修正内容

- 补齐 CuTe tensor、MMA、TMEM allocator、copy atom/traits 和 BF16 类型的头文件。
- 块尺寸和字节数改为 `constexpr`，供模板参数与 `__launch_bounds__` 使用。
- `CUTensorMap` 改为正确的 `CUtensorMap`；kernel 参数使用
  `const __grid_constant__`，使 TMA 指令获取 grid constant 描述符的地址。
- barrier 指针从 `shared + A_BYTES + B_BYTES` 构造。`shared_base` 是 shared
  地址空间的 32 位地址数值，不能将它直接强转为 API 接收的 C++ 指针。
- `allocation_address` 改为 `reinterpret_cast<uint32_t*>(mma_barrier + 1)`。
  `mma_barrier` 是 `uint64_t*`，`+1` 跨过 8B；原来的 `+8` 跨过 64B，超出 shared 分配。
- allocator 名字修正为 `Allocator1Sm`。
- 补齐 `descriptor(...)` 调用：A 从 `shared_base + step*32` 读，B 从
  `shared_base + A_BYTES + step*32` 读。一条 MMA 消耗 16 个 BF16，即 32B 的 K 偏移。
- `accumulate` 使用逻辑或 `||`，只有第一轮第一条 MMA 不累加旧值。
  原先对两个布尔条件使用 `|` 数值上也成立，改动让意图更明确。
- TMEM load 等待后加入 `__syncthreads()`，确保所有 warp 读完才释放 TMEM。
- 提前结束 `gemm` 的函数体，将 host 函数 `encode_tensor_map` 放到命名空间作用域。
- 展开固定次数的 MMA/写回循环；输出地址计算使用 `int64_t(row)`，避免 32 位乘法溢出。

## 2026-10-09 实测

北京时间 15:24，Modal NVIDIA B200（148 SM），CUDA 12.8.1 / nvcc 12.8.93，
cuBLAS 12.8.4，驱动 580.95.05，CUTLASS 4.2.1，固定 commit
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`。
编译选项为 `-std=c++17 -O3 -lineinfo -arch=sm_100a`，保留库的默认断言。

`M=N=K=4096`，三次 run 中位耗时的中位数：

| 实现 | 耗时 | 吞吐量 |
| --- | ---: | ---: |
| 修正后的 `write_by_myself2/v1.cu` | 0.486426 ms | 282.55 TFLOPS |
| 原版 PTX `cuda_kernel/v1_new.cu` | 0.487020 ms | 282.20 TFLOPS |
| cuBLAS（修正版三次 run 中的中位数） | 0.091584 ms | 1500.69 TFLOPS |

修正版相对 PTX 的耗时差为 -0.12%，小于本次测量波动，性能基本一致。
修正版耗时约为 cuBLAS 的 5.31 倍。

| run | 修正版耗时 | PTX 耗时 |
| --- | ---: | ---: |
| 1 | 0.486917 ms | 0.487634 ms |
| 2 | 0.485932 ms | 0.487020 ms |
| 3 | 0.486426 ms | 0.486225 ms |

复用 [原 benchmark](../cuda_kernel/benchmark.cu)，通过一个命名空间别名适配
`v1` 到 benchmark 所用的 `mojo_v1`。PTX 与修正版采用 AB/BA/AB 交替执行；
每次预热 10 次，CUDA Graph 每次 replay 执行 100 次 GEMM，CUDA events 计时，
custom/cuBLAS 交替测量 7 轮，取中位数。cuBLAS 使用 FP32 累加，禁用降低精度的归约。

正确性与检查：

- `64×256×64`、`128×512×192`、`192×256×512`、`256×768×1024` 和三次 `4096³`
  均通过 identity 精确对照、随机输入对照 cuBLAS，以及 32 个 CPU FP64 采样检查。
- 随机输入与 cuBLAS 的输出一致，`max_abs_error=0`，`relative_l2=0`。
- 在 `128×512×192` 上，memcheck/synccheck 均为 0 errors；racecheck 为
  0 hazards、0 errors、0 warnings。
- 修正版使用 146 registers，PTX 使用 140 registers；两者均为 0 stack、0 spill。
- [生成的 PTX](write_v1.sm100a.ptx) 中，结果读取仍是一条
  `tcgen05.ld.sync.aligned.16x256b.x32.b32`。
- 已核对本地算子、PTX 基线和 benchmark 的 SHA256 与远程运行使用的源码一致。

复现：

```bash
.venv/bin/modal run kernel/write_by_myself2/run_v1_modal.py --sanitize
# 指定尺寸
.venv/bin/modal run kernel/write_by_myself2/run_v1_modal.py --m 128 --n 512 --k 192
```

支持正的 M/N/K，且 `M%64=N%256=K%64=0`。

[完整数据和编译/检查日志](results_v1_b200_4096x4096x4096.json)；
[Modal 运行记录](https://modal.com/apps/akseven6221/main/ap-ST0Nr1F4ojBah1y0LY1HGr)。
