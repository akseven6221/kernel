# v21：rowsum 的小型 ILP

v20 pair-first max 没有明显收益，继续从 v16 出发。
只把单条 64-step pair-sum 链改为 4 条独立链，最后两层合并；maximum 与 exponent 生成保持不变。
相比 v10 的 8-chain max+sum，当前只有 sum 修改，且新的 384-thread/224-register 配额降低了状态压力。
加法顺序改变，因此检查所有普通/大/恒定超大 logits 和 causal/tail 输入后才记录性能。


## 最终统一复测

父版本：v16；决定：保留。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 17.814 | 10.640 | 59.73 |
| 1,1024,8,128 | False | 28.982 | 16.314 | 56.29 |
| 1,4096,8,128 | False | 95.718 | 53.875 | 56.29 |
| 1,8192,8,128 | False | 365.176 | 202.405 | 55.43 |
| 1,4096,8,128 | True | 96.373 | 53.230 | 55.23 |
| 1,8192,8,128 | True | 290.459 | 104.394 | 35.94 |
| 2,4096,16,128 | False | 377.840 | 214.342 | 56.73 |
| 2,4096,16,128 | True | 250.090 | 120.896 | 48.34 |
