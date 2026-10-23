# v5：CUDA v6 的分块双缓冲尾声 CuTe/CUTLASS 实现

[v5.cu](v5.cu) 将 [cuda_kernel/v6.cu](../cuda_kernel/v6.cu) 的主要 PTX 改成
CuTe/CUTLASS 接口，沿用本目录 v2/v3/v4 的改写方式，命名空间为 `v5`。
CUTLASS 固定为 4.2.1，commit `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`。

计算为行主序 `C[M,N] = A[M,K] × B[N,K]ᵀ`，BF16 输入/输出、FP32 累加。
每 CTA 192 线程，cluster `(2,1,1)`，两个 CTA 共同完成 `256×MMA_N` 输出。
支持正的 M/N/K，要求 `M%256=N%MMA_N=K%64=0`，没有尾块处理。

## 保留的流水线

与原 PTX 一样，两个 CTA 的 warp 4 各选一个线程加载本地 A/B；所有加载的
完成字节记到 rank 0 的 TMA-full barrier。rank 0 的 warp 5 等数据就绪后发 MMA，
每轮沿 K 发四条 `256×MMA_N×16` 的双 CTA MMA，并将 stage 完成通知 multicast
到双方的 MMA-empty barrier。最终 compute barrier 通知双方 warp 0–3 开始写结果。

producer 从 `{index=0, phase=1}` 开始，consumer 从 `{0,0}` 开始。
各自每绕完一圈才翻转 phase。保留原版 stage 数公式、shared 偏移、
128B swizzle 的 A/B，以及 512 列 TMEM 分配。

| MMA_N | 每 CTA 的 A/B tile | A/B stages | 输出缓冲 | 每 CTA 动态 shared |
| --- | --- | ---: | --- | ---: |
| 256（默认） | A 128×64，B 128×64 | 6 | 两份 128×32 BF16 | 213100 B |
| 128 | A 128×64，B 64×64 | 8 | 两份 128×32 BF16 | 213132 B |

与 PTX 基线共用编译宏 `MOJO_V6_MMA_N`，可设为 128 或 256。
host 端仍使用 CUDA Driver API 编码 tensor map。

## 分块输出

每次只从 TMEM 读 32 列，每线程 upper/lower 各 16 个 FP32 值，
对应 `Copy_Atom<SM100_TMEM_LOAD_16dp256b4x,float>`。
两次读后由 `fence_view_async_tmem_load()` 等待完成，再转 BF16，
经 `SM90_U32x4_STSM_N` 写入当前的 `128×32` shared 缓冲。
输出采用 **64B swizzle**，两个缓冲共 16384 B。

每轮写入 `(stage%2)` 号缓冲，所有输出线程执行 `tma_store_fence()` 后，
使用 `NamedBarrier::sync(128, ReservedNamedBarriers::EpilogueBarrier)` 汇合，
保持原版 barrier ID=1，只有 warp 0 的 lane 0 发 TMA store。

- 中间轮次 `wait_group 1`：最多保留一个在途 store。
- 从第 1 轮开始，到倒数第 2 轮，额外执行一次 128 线程 named barrier，
  将旧缓冲可复用的条件传给其余输出线程，再进入下一轮。
- 第 0 轮之后可直接写另一个缓冲；最后一轮 `wait_group 0` 排空全部 store。
- 所有线程最终参加 cluster 同步，双方 warp 0 协作释放 TMEM，再同步后退出。

## PTX 与接口对应

| 操作 | CuTe/CUTLASS 接口 |
| --- | --- |
| elect / cluster rank / cluster 同步 | `elect_one_sync()` / `block_rank_in_cluster()` / `cluster_sync()` |
| 双 CTA TMEM 分配、释放 | `cute::TMEM::Allocator2Sm` |
| 描述符预取 | `cute::prefetch_tma_descriptor()` |
| barrier 初始化 / 发布 / 等待 / 期待字节 | `ClusterBarrier` / `fence_barrier_init()` / `ClusterTransactionBarrier` |
| 双 CTA TMA load | `cute::SM100_TMA_2SM_LOAD_MULTICAST_2D` |
| shared / instruction 描述符 | `cute::UMMA::SmemDescriptor` / `make_runtime_instr_desc()` |
| 双 CTA MMA | `cute::SM100_MMA_F16BF16_2x1SM_SS` |
| stage 完成 / 最终 compute 通知 | `cutlass::arch::umma_arrive_multicast_2x1SM()` |
| 16 个 FP32 的 TMEM load / 等待 | `Copy_Atom<SM100_TMEM_LOAD_16dp256b4x,float>` / `fence_view_async_tmem_load()` |
| stmatrix | `cute::SM90_U32x4_STSM_N` |
| 输出线程汇合 | `cutlass::arch::NamedBarrier`，128 线程，ID=1 |
| 输出 proxy fence / TMA store / commit | `tma_store_fence()` / `SM90_TMA_STORE_2D` / `tma_store_arrive()` |

MMA 与完成通知封装内部执行 elect，由完整的 consumer warp 调用。
producer 每 CTA 选一个 lane 发出 TMA，mask 仍为 `1<<rank`，只写本地数据；
2SM 封装将完成通知路由到 rank 0。

保留两处 `tcgen05.fence::after_thread_sync`，以及完整的
`cp.async.bulk.wait_group 0/1`。CUTLASS 4.2.1 没有前者的对应封装，
`cute::tma_store_wait<N>()` 则只有 `.read` 等待，不能直接替代原版完整 store 等待。
接口依据固定版本官方
[TMEM copy 实现](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cute/arch/copy_sm100.hpp)、
[TMA store 实现](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cute/arch/copy_sm90_tma.hpp)、
[barrier 实现](https://github.com/NVIDIA/cutlass/blob/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8/include/cutlass/arch/barrier.h)。

## 2026-10-18 实测

Modal NVIDIA B200（148 SM），驱动 610.57.04，CUDA 12.8.1 / nvcc 12.8.93，
cuBLAS 12.8.4，Compute Sanitizer 2025.1.0.0。

- 256/128 两种配置各 11 组尺寸的原 PTX/CuTe identity、随机输入和 CPU FP64
  抽样检查均通过。全部随机检查与 cuBLAS 的 `max_abs_error=0`、`relative_l2=0`，
  包括主测尺寸计时前与计时后的检查。
- 两种配置的 memcheck、synccheck、racecheck 及额外大尺寸 racecheck
  全部通过，零错误、零警告。具体尺寸见下方复现说明。
- 两种配置均已核对本地 kernel、原 PTX 和 benchmark 的 SHA256 与远程文件一致。
- ptxas 的 CuTe/原 PTX 寄存器数分别为 47/31（256 配置）、49/32（128 配置），
  均为零 stack、零 spill。
- 256 配置生成的 PTX 包含 16 条展开的 `16x256b.x4` TMEM load、32 条
  `stmatrix.x4`、8 次 TMA store、7 次完整 `wait_group 1` 和最后一次
  `wait_group 0`；128 配置相应为 8/16/4/3/1，保留了分块双缓冲结构。

`4096×4096×4096`，三次 run 中位数的中位数：

| MMA_N | CuTe 耗时 | 原 PTX 耗时 | CuTe 相对 PTX 耗时变化 | 同场 cuBLAS 耗时 | CuTe/cuBLAS 吞吐量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.131843 ms | 0.121783 ms | +8.26% | 0.101044 ms | 76.64% |
| 128 | 0.195022 ms | 0.127324 ms | +53.17% | 0.088159 ms | 45.20% |

每行 PTX、CuTe、cuBLAS 在同一块 GPU 上交替测量，两种配置使用不同实例。
cuBLAS 列取 CuTe 三次运行中的 cuBLAS 中位数。未锁频，以上是本次测量结果。
128 配置的移植版有明显性能差距，原因尚未定位，没有将其归因于某个接口或
寄存器数量；本次完成接口移植与验证，未继续调参。

- 256：[原始结果及完整日志](results_v5_b200_4096x4096x4096.json)，
  [Modal 运行](https://modal.com/apps/akseven6221/main/ap-0qjchXPSyotrusBrYQkQVj)，
  [CuTe PTX](write_v5.sm100a.ptx)，[原版 PTX](ptx_v6.sm100a.ptx)。
- 128：[原始结果及完整日志](results_v5_b200_4096x4096x4096_mma128.json)，
  [Modal 运行](https://modal.com/apps/akseven6221/main/ap-Z79D8B7LNSpiMEhvy03pad)，
  [CuTe PTX](write_v5_mma128.sm100a.ptx)，[原版 PTX](ptx_v6_mma128.sm100a.ptx)。

## CuTe v4 / v5 同卡直接对比

2026-10-18，在同一块 Modal B200 上比较本目录的 CuTe v4/v5，
`M=N=K=4096`、`MMA_N=256`。复用同一份原 benchmark，只分别替换 include、
命名空间别名和结果标签，输入使用相同的固定随机种子。
执行六组配对，AB/BA 交替，两种实现各有三次先跑、三次后跑；
每次运行保持下方说明的预热、CUDA Graph 和七轮 CUDA events 计时。

| 实现 | 六次 run 中位数的中位数 |
| --- | ---: |
| CuTe v4 | 0.129935 ms |
| CuTe v5 | 0.132201 ms |

v5 的汇总耗时高 1.74%；六组配对的耗时变化中位数为 +1.41%，
各组范围为 -2.30% 到 +4.91%，其中一组 v5 更快。
这次没有看到 v5 提速，两者表现接近，不能据此断言 v5 在所有尺寸下稳定更慢。
此前不同 GPU 实例的版本结果不作为这次直接比较的依据。

十二次 GPU 端点采样的 UUID 均为
`GPU-15e6a31d-4269-0390-2b5d-f8b6526bdad3`，记录了频率、功率与温度，未锁频。
全部十二次运行的 identity、随机输入及 CPU FP64 抽样检查通过；
与 cuBLAS 的 `max_abs_error=0`、`relative_l2=0`，本地/远程源码 SHA256 一致。

[配对数据与完整日志](results_compare_v4_v5_b200_4096x4096x4096.json)，
[Modal 运行](https://modal.com/apps/akseven6221/main/ap-lBAr8sepFaGVpQy1clRIgK)。
复现：

```bash
.venv/bin/modal run kernel/write_by_myself2/run_compare_v4_v5_modal.py
```

## 复现

```bash
.venv/bin/modal run kernel/write_by_myself2/run_v5_modal.py --sanitize
.venv/bin/modal run kernel/write_by_myself2/run_v5_modal.py --mma-n 128 --sanitize
# 也可追加 --m M --n N --k K。
```

脚本使用 Modal B200、CUDA 12.8.1、`-std=c++17 -O3 -lineinfo -arch=sm_100a`，
保留默认编译断言。复用原 [benchmark_v6.cu](../cuda_kernel/benchmark_v6.cu)，
只替换 include、命名空间别名和结果标签。

每种配置分别运行 11 组尺寸，检查 identity 全量精确对比、随机输入全量对照
cuBLAS，以及每次 32 个 CPU FP64 点积抽样。K 覆盖不足一圈、恰好一圈、
首次复用、两圈前后和三圈之后。128 配置还覆盖 N=128/384，检查 N 不必被 256 整除。

`--sanitize` 未筛选 kernel，同时检查自定义算子和 cuBLAS。256 配置在
768×768×832 上运行 memcheck/synccheck/racecheck，在 4096×4096×1216 上额外
racecheck；128 配置对应 768×384×1088 和 4096×4096×1600。

两种实现同卡以 PTX/CuTe、CuTe/PTX、PTX/CuTe 的顺序测量。
每次直接预热 100 次，各捕获含 100 次 GEMM 的 CUDA Graph，交替预热各 50 次
replay，再用 CUDA events 交替测量 custom/cuBLAS 七轮取中位数；最终取三次
run 中位数的中位数。cuBLAS 为 BF16 输入/输出、FP32 累加，禁用 reduced-precision
reduction。不计编译、分配、传输、tensor map 编码和验证时间，未锁频。

结果保存源码 SHA256、编译信息、各项验证和 sanitizer 日志、逐轮计时，并检查
本地/远程源码一致。脚本导出 CuTe 和原版 PTX；128 配置文件名有 `_mma128` 后缀，
与 256 配置分别保存。
