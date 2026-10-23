# CUDA + PTX FlashAttention-4 实验

**学习入口：[从串行基线到 v51 的 23 课教程](tutorial/README.md)**。
按优化主题组织，包含中文导读源码、真实父版 diff、同机性能证据、同步约束和练习；
长序列与小 grid 两条路线最终汇合到 v51。

目标：把 [MLC FA4 章节](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/chapter_flash_attention/index.html#id5)
的 Blackwell attention forward 数据路径改写为直接 CUDA C++/inline PTX，并在 Modal B200 上与
[Dao-AILab 官方 FA4](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute) 同机比较。

已迭代到 v51，推荐 [v51](v51/README.md)：根据 shape、causal 和设备 SM 数选择单/双 Q-stage。
第二轮 v27–v51 共 25 个版本，23 个运行通过；v34 启动失败、v35 超时，源码和失败记录均保留。
有效版本均保存性能样本、NCU 原始报告与逐轮取舍。

同一 B200 统一复测，B=1、H=8、D=128、FP16：

| S | causal | 第一轮最佳长序列版 v24 us | v51 us | 官方 FA4 us | 官方吞吐 % |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | False | 16.946 | 12.038 | 10.731 | 89.14 |
| 1024 | False | 27.915 | 19.413 | 16.304 | 83.99 |
| 8192 | False | 362.376 | 276.085 | 203.664 | 73.77 |
| 8192 | True | 184.813 | 142.614 | 105.736 | 74.14 |

8192 长序列比同次运行的 v24 提速约 1.30–1.31×，尚未达到官方速度。
[v26](v26/README.md) 保留教程的四 WG、P split、mixed exponent 与 TMA output 数据路径，供阅读对照。

- [第二轮优化报告、全部 8 个形状和停止理由](OPTIMIZATION_ROUND2.md)
- [第一轮 v1–v26 历史报告](OPTIMIZATION_LOG.md)
- [全部历史实验及失败记录索引](MEASUREMENTS.md)
- [版本关系与源码 SHA256](versions.json)
- [同步协议与修正记录](SYNCHRONIZATION.md)
- [第二轮最终同机原始 benchmark](runs/r2_final_all/results.json)
- [v51 短序列 sanitizer](runs/r2_v51_sanitize_small/) / [长序列 sanitizer](runs/r2_v51_sanitize_large/)
- [第一轮原始 benchmark](runs/final_all_versions/results.json) / [第一轮 sanitizer](runs/final_sanitizers/)

约定范围：BSHD contiguous、FP16 输入/输出、FP32 累加、head_dim=128、self attention，
支持 non-causal/causal；不包含 backward、dropout、GQA、varlen、FP8。官方对照调用
`flash_attn.cute.flash_attn_func`（`flash-attn-4==4.0.0b35`），不使用 FA2/FA3 或 PyTorch SDPA 代替。

每个 `vN/kernel.cu` 是完整快照。`vN/README.md` 记录前版的瓶颈证据、单项修改、实测结果和决定。
`runs/<label>/` 保存完整运行环境、源码 SHA256、逐输入正确性结果、所有计时样本、nvcc/ptxas 日志、
PTX、SASS 以及 Nsight Compute 的原始 `.ncu-rep` 和文本/CSV。失败记录也保留，不计作通过。
有效版本的最终 `results.json` 和 `profile/` 已从各轮统一复测归档；当前源码、benchmark 快照、profile
快照 SHA256 均一致。历史实验快照可能早于同步或数值修正，以索引中的 hash 校验列区分。

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v24,v51 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v51 --mode profile
.venv/bin/modal run kernel/fa4/run_modal.py --versions v51 --mode sanitize --sanitize-seq 8191
.venv/bin/modal run kernel/fa4/run_modal.py --versions v51 --mode stress
```

计时：同一 B200、同一份 Q/K/V、相同 CUDA stream；双方都先 JIT/预热，然后用 20-node CUDA Graph
剔除 CPU launch 开销。9 轮交替顺序计时，保存原始样本并取中位数。吞吐量按
`4*B*H*S*S*D/time` 计算；causal 使用半矩阵 FLOP 近似。`official_throughput_percent` 是
`100*official_us/custom_us`，不是“相对官方的加速百分比”。NCU replay 时间只用于诊断，不用于排名。

正确性：每版本 40 项独立 FP32 matmul + softmax 参考检查（关闭 TF32），覆盖 causal/non-causal、
S=128/256/257/511、B=2/H=3，以及 normal、零 logits、大 logits、恒定超大 logits。
第一轮 26 版共 1040 项通过；第二轮 23 个有效新版本及 v1/v6/v24 基线共 1040 项通过。
8 个性能形状另补测与官方的差值。绝对/相对容差 0.003，同时保存 max_abs、RMSE、relative L2 及有限性。
v24/v25/v26 在 S=511 的两种 causal 模式上完成 memcheck/synccheck/racecheck，18 项检查全部通过。
v51 另通过 120 项多 seed/shape/scale 的 FP32 压力检查，以及 30-shape sweep；在 S=511 和 S=8191
上完成 causal/non-causal × memcheck/synccheck/racecheck，共 12 项全部通过，覆盖两条 dispatch 路径。
恒定 Q/K=30000 的压力案例中官方 b35 出现非有限输出，详情和自写 FMA 路径的修正见完整报告；该输入不用于性能排名。

复刻路径说明：v1 已使用 TMA、`tcgen05.mma`、TMEM 和 online softmax，但采用串行单 Q stage。
只有后续加入对应流水线机制的版本，才具有章节所述的角色分工/双 Q stage；不能把 v1 称为完整流水线复刻。

参考源码：网页给出的 `mlc-ai/tirx-kernels` 固定 revision 及 main raw URL 均无法取得（404/超时）；
网页正文已保存在 `reference/chapter.html` 和 `chapter.txt`。实现依据网页中的布局、伪代码与 PTX ISA，
不是用 TIRx 编译产物冒充 CUDA 源码。官方 README/interface 的下载快照保存在 `reference/`。
