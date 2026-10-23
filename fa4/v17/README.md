# v17：合并 TMEM score load

只把两条 x64 score load 合为一条 x128；仍然读取完整 128-score row，再显式 wait::ld。
v7 的 128-register entry 下这条指令曾编译失败；v16 的 entry=168 已高于最初报告的最小 146，
再次评估减少一次 collective load 的收益。保持 max ILP、FMA、mask fast path 和整个 pipeline 不变。
若 compiler register allocation 变差或没有实测收益，保留较快的 x64 版本。

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
| 1,512,8,128 | False | 17.922 | 10.640 | 59.37 |
| 1,1024,8,128 | False | 29.179 | 16.314 | 55.91 |
| 1,4096,8,128 | False | 97.613 | 53.875 | 55.19 |
| 1,8192,8,128 | False | 373.416 | 202.405 | 54.20 |
| 1,4096,8,128 | True | 96.280 | 53.230 | 55.29 |
| 1,8192,8,128 | True | 289.819 | 104.394 | 36.02 |
| 2,4096,16,128 | False | 386.301 | 214.342 | 55.49 |
| 2,4096,16,128 | True | 250.278 | 120.896 | 48.30 |
