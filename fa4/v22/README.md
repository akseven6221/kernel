# v22：causal 的 query-major reverse 调度

从 v16 出发，只改变 causal 的 CTA -> query/head/batch 映射；non-causal 保持原映射。
按 query tile 从后往前、同 query 下遍历全部 batch/head 的顺序启动，让重的 causal CTAs 先运行。

依据 `runs/v16_causal_profile`（S=8192）：SM active cycles 最大比均值高 41.38%，最小低 36.51%，
NCU 给出 distribution imbalance 23.86% 的估计。此前 causal 的 FLOP 近似只有 non-causal 一半，
实际 latency 却接近，说明尾部不均衡值得单独优化。
一个“各 SM 等速、一 CTA 占一 SM”的调度模型预测 makespan 从 196 到 127（任意工作单位）；
这是选取实验的推断，不是实际 speedup。以 event timing 与 causal NCU 的真实变化作决定。

同一 grid 大小，只重映射逻辑 tiles，不跳过合法 attention 工作。额外检查 B=2、H=3 的 causal/tail，
确保 flatten/remap 没有跨 batch/head 混淆。


## 最终统一复测

父版本：v16；决定：保留。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 17.605 | 10.640 | 60.44 |
| 1,1024,8,128 | False | 29.387 | 16.314 | 55.51 |
| 1,4096,8,128 | False | 100.696 | 53.875 | 53.50 |
| 1,8192,8,128 | False | 386.846 | 202.405 | 52.32 |
| 1,4096,8,128 | True | 103.112 | 53.230 | 51.62 |
| 1,8192,8,128 | True | 199.378 | 104.394 | 52.36 |
| 2,4096,16,128 | False | 394.741 | 214.342 | 54.30 |
| 2,4096,16,128 | True | 205.469 | 120.896 | 58.84 |
