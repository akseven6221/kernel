# v15：重新评估较小的 max ILP

v14 的 FMA 在 non-causal 提升约 4%，causal 略有回退。以 v14 为起点，
只把 maximum reduction 改为 4 条独立链 + 两层合并；psum 保持原顺序。
这是对 v10 失败原因的收缩实验：v10 同时使用 8 条 max/sum chains，扩大了编译器状态和 spill。
v12-v14 已去掉 correction mailbox 并增加初始 register budget；此处检验较小的 max ILP 是否有收益。
若没有稳定收益，不把理论依赖链缩短当作性能提升。

最后数值压力检查加入 Q/K=30000 的恒定大 logits，发现 FMA reference rounding 的公共偏移
会导致 FP16 P 溢出（最小复现 `runs/v19_extreme_before_fix`）。最终源码在 `abs(reference)>65536`
时走先减再乘的稳定路径，让当前 maximum 的指数输入精确为 0；普通输入仍用 FMA fast path。
最终统一 comparison / correctness 重测这个修正。


## 最终统一复测

父版本：v14；决定：保留。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 18.526 | 10.640 | 57.43 |
| 1,1024,8,128 | False | 30.112 | 16.314 | 54.18 |
| 1,4096,8,128 | False | 100.661 | 53.875 | 53.52 |
| 1,8192,8,128 | False | 384.298 | 202.405 | 52.67 |
| 1,4096,8,128 | True | 102.267 | 53.230 | 52.05 |
| 1,8192,8,128 | True | 308.850 | 104.394 | 33.80 |
| 2,4096,16,128 | False | 395.226 | 214.342 | 54.23 |
| 2,4096,16,128 | True | 264.776 | 120.896 | 45.66 |
