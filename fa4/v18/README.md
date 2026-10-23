# v18：单 K/V slot 的资源对照实验

只把 K/V pipeline depth 从 2 改为 1，SMEM 从约 193 KiB 降至 129 KiB；MMA/softmax、
query stages、register budget 和数学仍为 v16。Loader 必须等当前 PV 完成才重用唯一的 slot。
这是对 shared-memory/L1 压力与搬运重叠之间 tradeoff 的实测，预计可能因失去预取而回退，
但需要验证较低 shared-memory carveout 能否补偿 local-memory 等待。保留无论快慢的实测与 profile。

最后数值压力检查加入 Q/K=30000 的恒定大 logits，发现 FMA reference rounding 的公共偏移
会导致 FP16 P 溢出（最小复现 `runs/v19_extreme_before_fix`）。最终源码在 `abs(reference)>65536`
时走先减再乘的稳定路径，让当前 maximum 的指数输入精确为 0；普通输入仍用 FMA fast path。
最终统一 comparison / correctness 重测这个修正。


## 最终统一复测

父版本：v16；决定：回退。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 19.755 | 10.640 | 53.86 |
| 1,1024,8,128 | False | 32.949 | 16.314 | 49.51 |
| 1,4096,8,128 | False | 112.131 | 53.875 | 48.05 |
| 1,8192,8,128 | False | 436.966 | 202.405 | 46.32 |
| 1,4096,8,128 | True | 114.784 | 53.230 | 46.37 |
| 1,8192,8,128 | True | 351.520 | 104.394 | 29.70 |
| 2,4096,16,128 | False | 488.574 | 214.342 | 43.87 |
| 2,4096,16,128 | True | 317.712 | 120.896 | 38.05 |
