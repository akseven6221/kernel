# v23：将 causal 调度应用到 rowsum ILP 版本

在 v21 上仅加入 v22 的 causal tile remap。
v21 rowsum ILP 提升约 4-6%；v22 的 query-major reverse 在 S=8192 causal 从约 309 到 200 us，
与 NCU 的尾部不均衡证据吻合。此版组合两个已经分开验证的收益，non-causal 仍与 v21 使用同样映射。
40-case 正确性检查包含 B=2/H=3/tail；最终同机 comparison 重新测组合版本，不能直接相乘历史 speedup。


## 最终统一复测

父版本：v21；决定：保留。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 16.885 | 10.640 | 63.02 |
| 1,1024,8,128 | False | 28.293 | 16.314 | 57.66 |
| 1,4096,8,128 | False | 96.373 | 53.875 | 55.90 |
| 1,8192,8,128 | False | 369.352 | 202.405 | 54.80 |
| 1,4096,8,128 | True | 96.573 | 53.230 | 55.12 |
| 1,8192,8,128 | True | 185.139 | 104.394 | 56.39 |
| 2,4096,16,128 | False | 376.984 | 214.342 | 56.86 |
| 2,4096,16,128 | True | 194.090 | 120.896 | 62.29 |
