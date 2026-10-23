# v4：CUDA v5 的五级流水线 CuTe/CUTLASS 实现

[v4.cu](v4.cu) 按本目录 [v2.cu](v2.cu)、[v3.cu](v3.cu) 的方式，
将 [cuda_kernel/v5.cu](../cuda_kernel/v5.cu) 的主要 PTX 换成 CuTe/CUTLASS 接口。
命名空间为 `v4`；使用 CUTLASS 4.2.1，固定 commit
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`。

计算为行主序 `C[M,N] = A[M,K] × B[N,K]ᵀ`，BF16 输入/输出、FP32 累加。
保留原版 `(2,1,1)` cluster、每 CTA 192 线程、512 列 TMEM、五级 A/B 环形缓冲，
以及完整 `128×256` 的输出共享缓冲。两个 CTA 共同计算 `256×256` 输出。
每 CTA 动态共享内存为 229536 B，支持正的 M/N/K，要求
`M%256=N%256=K%64=0`，不处理尾块。

## 流水线与 warp 分工

| 执行者 | 工作 |
| --- | --- |
| 两个 CTA 的 warp 4，各一个 elected lane | 等待本地 MMA-empty barrier；加载本地 A/B，完成字节统一记到 rank 0 的 TMA-full barrier |
| rank 0 的 warp 5 | 等待 TMA-full；每轮发四条 2SM MMA；完成通知 multicast 到双方对应 stage 的 MMA-empty barrier |
| 两个 CTA 的 warp 0–3 | 等待最终 compute barrier，读 TMEM、转 BF16、stmatrix、TMA 写回 |
| rank 1 的 warp 5 | 不参与主循环，参加收尾 cluster 同步 |

每个 stage 有独立的 TMA-full/MMA-empty barrier。producer 从 `{index=0, phase=1}`
开始，允许首次直接使用空槽；consumer 从 `{0,0}` 开始，等待数据就绪。
两者独立推进，每经过五个 stage 环绕一次才翻转 phase。
不使用 CUTLASS 高层 GEMM 或 `PipelineTmaUmmaAsync` 重写调度，便于逐段对照原版。

## 主要接口对应

| 操作 | CuTe/CUTLASS 接口 |
| --- | --- |
| elect / cluster rank / cluster sync | `cute::elect_one_sync()` / `block_rank_in_cluster()` / `cluster_sync()` |
| TMEM 分配、释放许可、释放 | `cute::TMEM::Allocator2Sm` |
| tensor map 预取 | `cute::prefetch_tma_descriptor()` |
| barrier 初始化 / 发布 / 等待 | `ClusterBarrier` / `fence_barrier_init()` |
| leader 设置 TMA 期待字节数 | `ClusterTransactionBarrier::arrive_and_expect_tx()` |
| 2SM TMA load | `cute::SM100_TMA_2SM_LOAD_MULTICAST_2D` |
| MMA 描述符 | `cute::UMMA::SmemDescriptor` / `make_runtime_instr_desc()` |
| 双 CTA MMA | `cute::SM100_MMA_F16BF16_2x1SM_SS` |
| stage 释放 / 最终 compute 通知 | `cutlass::arch::umma_arrive_multicast_2x1SM()`，mask=3 |
| TMEM load / wait | `Copy_Atom<SM100_TMEM_LOAD_16dp256b32x,float>` / `fence_view_async_tmem_load()` |
| stmatrix | `cute::SM90_U32x4_STSM_N` |
| 仅输出线程同步 | `NamedBarrier::sync(128, ReservedNamedBarriers::EpilogueBarrier)` |
| 输出 proxy fence / TMA store / commit | `cute::tma_store_fence()` / `SM90_TMA_STORE_2D` / `tma_store_arrive()` |

`Mma::fma()` 和 `umma_arrive_multicast_2x1SM()` 内部已有 elect，必须由整个
consumer warp 调用；外层不再使用单 lane 条件。producer 仍由每 CTA 的一个 lane
发 TMA，mask 为 `1<<rank`，数据只进入自己的 shared；2SM 封装将完成通知路由到
rank 0 的 barrier，每轮共期待 65536 B。

输出 named barrier 只等待 warp 0–3 的 128 个线程；使用枚举重载保留原版
`bar.sync 1,128` 的 ID。整数重载会额外加上 CUTLASS 的保留 ID 偏移。

与 v2/v3 一样，保留两处 `tcgen05.fence::after_thread_sync` 和一处完整的
`cp.async.bulk.wait_group 0`。CUTLASS 4.2.1 没有前者的对应封装，
`cute::tma_store_wait<0>()` 只等待 shared 源读取完成（`.read`），不替代原版完整等待。

接口依据固定版本的官方
[barrier 实现](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cutlass/arch/barrier.h)、
[2SM TMA 实现](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cute/arch/copy_sm100_tma.hpp)、
[MMA 实现](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cute/arch/mma_sm100_umma.hpp)。

## 2026-10-18 实测

Modal NVIDIA B200，148 SM，驱动 610.57.04，CUDA 12.8.1 / nvcc 12.8.93，
cuBLAS 12.8.4，Compute Sanitizer 2025.1.0.0。默认编译断言保持开启。

- 11 组尺寸的原 PTX/CuTe identity、随机输入、CPU FP64 抽样检查均通过。
  全部随机检查与 cuBLAS 的 `max_abs_error=0`、`relative_l2=0`，
  包括主测尺寸三次计时前和计时后的检查。
- 768×512×576 的 memcheck/synccheck/racecheck 和
  4096×4096×1216 的额外 racecheck 均通过，零错误、零警告。
- ptxas：CuTe 使用 241 registers，原 PTX 使用 231 registers；均为零 stack、零 spill。
- 生成的 CuTe PTX 包含两条 TMEM load、32 条展开的 `stmatrix.x4`、
  `bar.sync 1,128`、完整 bulk wait，以及双 CTA MMA/commit 和 cluster 同步。
- 已核对本地 kernel、原 PTX 和 benchmark 的 SHA256 与远程编译文件一致。

`4096×4096×4096` 同卡交替测量，三次 run 中位数的中位数：

| 实现 | 耗时 | 吞吐量 |
| --- | ---: | ---: |
| CuTe v4 | 0.127683 ms | 1076.41 TFLOPS |
| 原 PTX v5 | 0.121333 ms | 1132.74 TFLOPS |
| cuBLAS（CuTe 三次运行中的中位数） | 0.098776 ms | 1391.42 TFLOPS |

本次 CuTe 耗时比原 PTX 高 5.23%，吞吐量为同场 cuBLAS 的 77.36%。
CuTe 三次耗时分别为 0.129208 / 0.127595 / 0.127683 ms；原 PTX 为
0.124036 / 0.121293 / 0.121333 ms。未锁频，以上是本次测量结果，
没有将差异归因于某个接口或寄存器数量，也没有进一步调参。

[完整结果与日志](results_v4_b200_4096x4096x4096.json)，
[Modal 运行](https://modal.com/apps/akseven6221/main/ap-996pfDvTjtvmGhFGRwSD2S)，
[CuTe 生成的 PTX](write_v4.sm100a.ptx)，[原版生成的 PTX](ptx_v5.sm100a.ptx)。

## 复现

在仓库根目录运行：

```bash
.venv/bin/modal run kernel/write_by_myself2/run_v4_modal.py --sanitize
# 指定其他满足整除条件的尺寸：
.venv/bin/modal run kernel/write_by_myself2/run_v4_modal.py --m 4096 --n 2560 --k 8192
```

脚本使用 Modal B200、CUDA 12.8.1、`-std=c++17 -O3 -lineinfo -arch=sm_100a`。
复用原 [benchmark_v5.cu](../cuda_kernel/benchmark_v5.cu)，只替换 include、命名空间
别名和结果标签。原 PTX、CuTe 均检查 identity 全量精确比较、随机输入全量对比
cuBLAS 和每次 32 个 CPU FP64 点积抽样。覆盖 K=64/128/192/256/320/384/576/640/
704/1216，包含未填满、恰好填满、第一次复用，以及多次 phase 环绕。

`--sanitize` 对 CuTe 实现运行未筛选的 memcheck/synccheck/racecheck
（768×512×576），额外对 4096×4096×1216 运行 racecheck。
两种 kernel 在同一 GPU 上按 PTX/CuTe、CuTe/PTX、PTX/CuTe 顺序计时，
每次均同场测 cuBLAS。保留原 benchmark 的预热、CUDA Graph 和 CUDA events 计时，
最终取三次 run 中位数的中位数。未锁频。

脚本保存源码 SHA256、编译信息、正确性及 sanitizer 日志、逐轮计时，
并核对本地/远程源码一致，导出原 PTX 和 CuTe 两份生成的 PTX。
