# v1_new 的 Modal B200 benchmark

2026-10-05，NVIDIA B200（148 SM），CUDA 12.8.1，cuBLAS 12.8.4，
驱动 580.95.05；`M=N=K=4096`，BF16 输入/输出、FP32 累加。

| 实现 | 中位耗时 | 吞吐量 | 相对同场 cuBLAS 吞吐量 |
| --- | ---: | ---: | ---: |
| v1_new.cu | 0.486488 ms | 282.51 TFLOPS | 18.90% |
| 原版 v1.cu（同一实例重跑） | 0.902845 ms | 152.23 TFLOPS | 10.20% |
| cuBLAS（v1_new 的基线） | 0.091932 ms | 1495.00 TFLOPS | 100% |

本次 `v1_new` 相对原版 `v1` 加速 **1.856 倍**，耗时约为其 cuBLAS 基线的
**5.292 倍**。原版 v1 配套测得的 cuBLAS 为 0.092080 ms、1492.60 TFLOPS。
两版在同一个 B200 容器中先后测量；这是本次 4096³ 的实测比较。
v1_new 的 7 轮耗时范围为 0.483880–0.500169 ms。

使用原版 [benchmark.cu](benchmark.cu) 的同一套测试逻辑，仅通过编译宏选择
`v1_new.cu` 或 `v1.cu` 并切换输出标签；没有修改两个核。
计时、随机 seed、cuBLAS 配置及误差阈值均沿用 [v1 比较口径](README.md#比较口径)：
直接预热 10 次，再预热 CUDA Graph；每个 graph 含 100 次 GEMM，
CUDA events 计时，交替测 custom/cuBLAS，共 7 轮，报告每次 GEMM 的中位耗时。

正确性和检查结果：

- v1_new 在 `64×256×64`、`128×512×192`、`192×256×512` 和 `4096³` 全部通过。
- identity 输入逐元素精确匹配；随机输入与 cuBLAS 的 BF16 输出完全一致，
  `max_abs_error=0`、`relative_l2=0`，32 个 CPU FP64 参考点也通过。
- 原版 v1 的 `4096³` 正确性检查通过，同样与 cuBLAS 输出完全一致。
- v1_new 在 `128×512×192` 上的 memcheck、synccheck、racecheck 全部零错误、零警告。

复现命令（仓库根目录）：

```bash
.venv/bin/modal run kernel/cuda_kernel/run_v1_new_modal.py --sanitize
```

脚本使用与原版相同的 CUDA 镜像及 `sm_100a` 编译参数，并在同一实例重跑原版 v1。
完整日志、7 轮时间、环境版本、源码 SHA256 和同机原版数据保存在
[results_v1_new_b200_4096x4096x4096.json](results_v1_new_b200_4096x4096x4096.json)。
运行记录见 [Modal app](https://modal.com/apps/akseven6221/main/ap-Onft5W10gavQ0DHBRoyDcJ)。
