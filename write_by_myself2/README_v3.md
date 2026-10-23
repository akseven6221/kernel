# v3：CUDA v4 的双 CTA CuTe/CUTLASS 实现

[v3.cu](v3.cu) 按 [v2.cu](v2.cu) 的改写方式，将
[cuda_kernel/v4.cu](../cuda_kernel/v4.cu) 的主要 PTX 换成 CuTe/CUTLASS 接口。
命名空间为 `v3`，原 PTX v4 保留作为同场性能基线。

计算为行主序 `C[M,N] = A[M,K] × B[N,K]ᵀ`，BF16 输入/输出、FP32 累加。
保留 `(2,1,1)` cluster、每 CTA 128 线程、单缓冲和原有 shared/TMEM 布局。
两个 CTA 共同完成 `256×256` 输出，每个 CTA 加载 `128×64` 的 A 和 B，
写回 `128×256` 输出。支持正的 M/N/K，要求 `M%256=N%256=K%64=0`。

## 指令对应

| 原 PTX / 操作 | CuTe/CUTLASS 接口 |
| --- | --- |
| `elect.sync` | `cute::elect_one_sync()` |
| cluster rank / cluster barrier | `cute::block_rank_in_cluster()` / `cute::cluster_sync()` |
| 双 CTA TMEM alloc / relinquish / dealloc | `cute::TMEM::Allocator2Sm` |
| `mbarrier.init` / expect_tx / wait / init fence | `cutlass::arch::ClusterBarrier`、`ClusterTransactionBarrier`、`fence_barrier_init()` |
| 双 CTA multicast TMA load | `cute::SM100_TMA_2SM_LOAD_MULTICAST_2D` |
| MMA shared descriptor / instruction descriptor | `cute::UMMA::SmemDescriptor` / `make_runtime_instr_desc()` |
| `tcgen05.mma.cta_group::2.kind::f16` | `cute::SM100_MMA_F16BF16_2x1SM_SS` |
| 双 CTA MMA 完成通知，mask=3 | `cutlass::arch::umma_arrive_multicast_2x1SM()` |
| `tcgen05.ld...16x256b.x32` | `cute::Copy_Atom<SM100_TMEM_LOAD_16dp256b32x,float>` |
| TMEM load wait | `cutlass::arch::fence_view_async_tmem_load()` |
| `stmatrix.m8n8.x4` | `cute::SM90_U32x4_STSM_N` |
| async proxy fence / TMA store / bulk commit | `cute::tma_store_fence()` / `SM90_TMA_STORE_2D` / `tma_store_arrive()` |

2SM TMA load 封装会清除 barrier 地址的 peer bit，使两个 CTA 的完成字节均记到
CTA 0 的 barrier，替代原版 `mapa(...,0)`。每 CTA 的 multicast mask 仍为
`1<<rank`，leader 每轮期待 `2*(A_BYTES+B_BYTES)=65536` 字节。
仅 leader CTA 的 warp 0 调用 MMA；MMA 与完成通知封装内部选择一个 lane 发射。

与 v2 一样保留三处内联 PTX：两处 `tcgen05.fence::after_thread_sync`，以及一处
完整的 `cp.async.bulk.wait_group 0`。CUTLASS 4.2.1 没有前者的对应封装；
`cute::tma_store_wait<0>()` 仅等待 shared 源读取完成（`.read`），因此后者保留原版
等待整个 store 完成的语义。没有把完整等待替换成 `.read`。

实现依据固定版本的官方
[2SM TMA 接口](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cute/arch/copy_sm100_tma.hpp)、
[MMA 接口](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cute/arch/mma_sm100_umma.hpp)、
[barrier 接口](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cutlass/arch/barrier.h)。

## 正确性与生成代码

2026-10-16，Modal NVIDIA B200，148 SM，CUDA 12.8.1 / nvcc 12.8.93，
cuBLAS 12.8.4，驱动 610.57.04。CUTLASS 4.2.1，固定 commit
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`。
编译参数 `-std=c++17 -O3 -lineinfo -arch=sm_100a`，保留默认断言。

- `256×256×64`、`512×512×192`、`768×768×576`、`768×512×576`、
  `512×768×128`、`4096×4096×64`：CuTe v3 与原 PTX v4 都通过 identity 全量精确比较、
  随机输入全量对比 cuBLAS，以及每次 32 个 CPU FP64 点积抽样检查。
- `768×512×576` 通过 memcheck、synccheck、racecheck；额外
  `4096×4096×576` racecheck 也无错误、无警告。
- 主测尺寸每次计时前、计时后均对照 cuBLAS 检查输出。
- 上述正确性检查和两个主测尺寸的全部随机检查，均为
  `max_abs_error=0`、`relative_l2=0`。
- ptxas：CuTe v3 使用 **229 registers**，原 PTX v4 使用 **235 registers**；
  两者均为 **0 stack / 0 spill**，每 CTA 动态 shared 均为 **98368B**。
- [生成的 CuTe PTX](write_v3.sm100a.ptx) 保留 `cta_group::2`，有两条展开的
  `16x256b.x32` TMEM load 和 32 条展开的 `stmatrix.x4`，以及 TMA store、
  proxy fence、完整 bulk wait 和 cluster 同步。
  [原版生成的 PTX](ptx_v4.sm100a.ptx) 一并保存。
- 本地 kernel、原 PTX 源文件和 benchmark 的 SHA256 与远程实际编译文件匹配。

## 性能与复现

复用原 [benchmark_v4.cu](../cuda_kernel/benchmark_v4.cu)，只替换 include、
命名空间别名和结果标签；其验证、cuBLAS 配置和计时逻辑均保持不变。
每个尺寸在同一 GPU 上按 PTX/CuTe、CuTe/PTX、PTX/CuTe 顺序测量，
每个实现的每次运行都同场测 cuBLAS。表中取三次 run 中位数的中位数，
cuBLAS 列取 CuTe 三次运行中的 cuBLAS 中位数。

每次运行直接预热 100 次，交替预热各 50 次 CUDA Graph replay，
每个 replay 包含 100 次 GEMM；CUDA events 测量 7 轮，交替 custom/cuBLAS
先后顺序取中位数。不计编译、分配、传输、tensor map 编码和验证时间。
cuBLAS 为 `cublasGemmEx`、`CUBLAS_COMPUTE_32F`，禁用 reduced-precision reduction；
未做 cuBLASLt 算法搜索。吞吐量按 `2*M*N*K/time` 计算。

| M×N×K | CuTe v3 | CuTe TFLOPS | 原 PTX v4 | cuBLAS | cuBLAS TFLOPS | CuTe/cuBLAS 吞吐量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096×4096×4096 | 0.342212 ms | 401.62 | 0.351108 ms | 0.089784 ms | 1530.77 | 26.24% |
| 4096×2560×8192 | 0.510370 ms | 336.62 | 0.529426 ms | 0.116925 ms | 1469.31 | 22.91% |

`4096³` 中，CuTe 比原 PTX 耗时低 2.53%；CuTe 三次结果为
0.342212 / 0.341548 / 0.345125 ms，PTX 为 0.351108 / 0.351911 / 0.349629 ms。
`4096×2560×8192` 中，CuTe 比原 PTX 耗时低 3.60%；CuTe 三次结果为
0.510886 / 0.510370 / 0.510084 ms，PTX 为 0.532900 / 0.529426 / 0.523497 ms。
两个尺寸分别使用不同的 Modal GPU 实例；每个尺寸内部的 PTX、CuTe、cuBLAS
在同一 GPU 上交替测量。这是本次运行中的差异，未锁频，不据此归因于某个封装
或寄存器数量变化。

运行：

```bash
.venv/bin/modal run kernel/write_by_myself2/run_v3_modal.py --sanitize
.venv/bin/modal run kernel/write_by_myself2/run_v3_modal.py --m 4096 --n 2560 --k 8192
```

启动前会检查尺寸约束；脚本使用 B200，保存源码 SHA256、编译信息、正确性与
sanitizer 日志、每轮计时和生成的 PTX。

- [4096³ 原始结果](results_v3_b200_4096x4096x4096.json)，
  [Modal 运行](https://modal.com/apps/akseven6221/main/ap-VWVxRRgj8EtD5bXe0oj75G)。
- [4096×2560×8192 原始结果](results_v3_b200_4096x2560x8192.json)，
  [Modal 运行](https://modal.com/apps/akseven6221/main/ap-9L6o1NcDD488SuYdJCHYUa)。
