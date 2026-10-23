# Profiling 驱动的优化记录

本文保留第一轮 v1–v26 的数据和结论。后续已完成 v27–v51，当前推荐 v51；见
[第二轮优化报告](OPTIMIZATION_ROUND2.md)，其中 v24 已与新版本同机重新计时。

所有最终版本在同一 Modal B200 上统一复测，FP16 输入/输出、FP32 accum、D=128。当前源码全部与运行快照 SHA256 匹配。
CUDA Graph：20 nodes / replay，10 轮预热，9 轮交替顺序，CUDA events 中位数；共 1040 项 FP32 正确性检查、208 个 version/shape 性能结果。

## 最终主结果

| Shape B,S,H,D | causal | v1 us | v24 us | 官方 FA4 us | v1→v24 | v24 官方吞吐 % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1,512,8,128 | False | 31.840 | 16.784 | 10.640 | 1.897× | 63.39 |
| 1,1024,8,128 | False | 43.989 | 27.776 | 16.314 | 1.584× | 58.73 |
| 1,4096,8,128 | False | 228.646 | 94.246 | 53.875 | 2.426× | 57.16 |
| 1,8192,8,128 | False | 829.216 | 360.952 | 202.405 | 2.297× | 56.08 |
| 1,4096,8,128 | True | 216.688 | 94.755 | 53.230 | 2.287× | 56.18 |
| 1,8192,8,128 | True | 612.062 | 181.642 | 104.394 | 3.370× | 57.47 |
| 2,4096,16,128 | False | 897.048 | 370.965 | 214.342 | 2.418× | 57.78 |
| 2,4096,16,128 | True | 638.728 | 191.168 | 120.896 | 3.341× | 63.24 |

长序列主要推荐 v24；512/1024 短序列 v6 更快。需要阅读四 WG / split P / mixed exponent / TMA store 的教程路径，查看 v26，不能把版本号当作性能排名。

## 每轮实验与取舍

| Version | Parent | 修改 | 决定 |
| --- | --- | --- | --- |
| [v1](v1/README.md) | — | 串行 TMA/TC/TMEM 基线 | 基线 |
| [v2](v2/README.md) | v1 | 直接 PTX exp2 | 保留 |
| [v3](v3/README.md) | v2 | 合并 output stores | 保留 |
| [v4](v4/README.md) | v3 | conditional rescaling | 小幅收益 |
| [v5](v5/README.md) | v4 | 每行一次 reciprocal | 保留 |
| [v6](v6/README.md) | v5 | 双 K/V buffer 预取 | 短序列候选 |
| [v7](v7/README.md) | v5 | 双 Q / 四 WG roles | 教程结构对照 |
| [v8](v8/README.md) | v7 | 50% polynomial exponent | 回退，保留实验 |
| [v9](v9/README.md) | v7 | warp arrivals | 无稳定收益 |
| [v10](v10/README.md) | v7 | 8 条 max/sum chains | 回退 |
| [v11](v11/README.md) | v7 | P split completion | 无稳定收益 |
| [v12](v12/README.md) | v7 | correction / epilogue 合并 | 保留 |
| [v13](v13/README.md) | v12 | valid-tile mask fast path | 保留 |
| [v14](v14/README.md) | v13 | FMA argument + 极大 reference 稳定分支 | 保留 |
| [v15](v15/README.md) | v14 | 4 条 max chains | 保留 |
| [v16](v16/README.md) | v15 | 168/224 register budget | 保留 |
| [v17](v17/README.md) | v16 | 单条 x128 score load | 回退 |
| [v18](v18/README.md) | v16 | 单 KV slot | 回退 |
| [v19](v19/README.md) | v16 | packed FMA/add | non-causal 回退 |
| [v20](v20/README.md) | v16 | pair-first max3 | 无稳定收益 |
| [v21](v21/README.md) | v16 | 4 条 rowsum chains | 保留 |
| [v22](v22/README.md) | v16 | causal query-major reverse | 保留 |
| [v23](v23/README.md) | v21 | 加入 causal reverse | 保留 |
| [v24](v24/README.md) | v23 | K/V readiness / lifetime 分离 | 主要推荐 |
| [v25](v25/README.md) | v24 | 5-slot K/V ring | 无额外收益 |
| [v26](v26/README.md) | v11 | 四 WG / P split / mixed exp / TMA output 整合 | 教程 non-causal 路径对照 |

## 关键 profile 证据

v1 的 output stores 有 3,932,160 excessive sectors，占 4,194,304 sectors 的约 94%；合并输出后降为 0。
PTX/SASS 显示每行 128 次 div.rn，以及通用 exp2f 的范围处理；改为直接 exp2 与一次 reciprocal 后去掉不必要指令。
完整 tile 的 mask fast path 去掉重复 bounds/select；4-chain maximum 与 rowsum 的较小 ILP 在新 register budget 下有效，而旧 8-chain 同时修改两种 reduction 没有收益。
v16 causal S=8192 的 SM active cycles 最大高于均值 41.38%、最小低 36.51%；v22 query-major reverse 后降为 +8.35% / −4.36%。该改动的同机 causal latency 从约 309 降到 200 us。
K/V 生命周期分离能小幅改善 overlap；再增加到 5-slot ring 没有额外稳定收益。Mixed polynomial、P split、warp arrivals、x128 load 均保留，但不能因技术来自论文就宣称在本实现里加速。

## 正确性与同步

40-case / version：B=1 的 S=128/256/257/511/H=2，以及 B=2/S=257/H=3；causal/non-causal；normal、zero、large logits、constant large logits。独立 FP32 matmul/softmax 参考关闭 TF32；保存 max_abs/RMSE/relative L2。性能形状另对照官方输出。
v24/v25/v26 在 S=511、causal/non-causal 上完成 memcheck/synccheck/racecheck：18 项检查均 exit 0，零错误和零警告，覆盖 KV ring reuse 与 TMA output 的边界。v6/v7/v12/v13 的早期审计也保留。
v12 首轮 causal 的 future-parity 等待失败已保留，修正为独立 FINAL completion。v1-v8 补齐 TC before-thread-sync fence。
恒定超大 Q/K=30000 暴露了 FMA 公共 reference rounding shift 导致 FP16 P 溢出。FMA 版本现有 abs(reference)>65536 的先减再乘稳定分支。本次官方 b35 在该压力案例的记录中出现非有限输出（260 条重复 version 对照记录）；它们不用于性能排名。官方在普通、零、large_logits=6 的对照全部通过。

## 本轮停止点与限制

最终 v24 相比更深 ring v25 和整合版 v26 更快。最后对 polynomial、packed arithmetic、max3、额外 buffer 与 TMA load 形状的尝试没有找到全面、可复现的下一项加速；本轮停止在这个经过验证的实现。没有证明全局最优，也没有达到官方 FA4 的速度。
范围仅 dense self-attention FP16 forward、head_dim=128、contiguous BSHD；没有 backward/dropout/GQA/varlen/FP8。v26 对照教程重点讲解的 non-causal 四角色数据路径；其 causal epilogue 仍放在 WG2，与文章 specialization 不同。
剩余主要差距包括 compiler register/spill、softmax/TMEM 调度和数据搬运布局。更大 MMA tile、多 CTA 或重新组织输入 TMA 布局都需要新的布局与同步设计，不能把未测的推断写成已完成的优化。

## 原始运行与重现

[同机总复测](https://modal.com/apps/akseven6221/main/ap-qLPHdS3TzXQPlq4jMmbAwz)；[最终 sanitizer](https://modal.com/apps/akseven6221/main/ap-oedfnwyHZRKG1gpfdKJ1e5)。
第一次全量 final profile 回传因网络 Connection lost 中断，失败日志保留；最终报告来自小批次重新运行，逐版本复制到 profile/，完整批次位于 runs/。

```bash
.venv/bin/modal run kernel/fa4/run_modal.py --versions v24,v26 --mode benchmark
.venv/bin/modal run kernel/fa4/run_modal.py --versions v24 --mode profile
.venv/bin/modal run kernel/fa4/run_modal.py --versions v24 --mode profile --seq 8192 --causal
.venv/bin/modal run kernel/fa4/run_modal.py --versions v24,v26 --mode sanitize
.venv/bin/python kernel/fa4/summarize.py
```
