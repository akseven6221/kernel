# v20：pair-first / 三输入 maximum reduction

v19 的 packed FMA/add 只在 causal 略有收益，主 non-causal 回退；继续使用 v16 作起点。
官方 `utils.fmax_reduce` 强制三输入 max 的思路提示检查自己的 max critical path。
本版先对相邻 score pair 求 max，再与 4 条 running-max chains 合并；inline PTX 明确 pair-first 顺序，
检查 ptxas 是否能使用 SM100 的三输入 max。即使没有融合，也让 pair 的 max 不依赖 running maximum。
计算的是相同 maximum，不改变 rowsum、指数、TMA/PV 或同步。

继承最后加入的超大 reference 稳定路径，避免 FMA 公共偏移令 FP16 P 溢出；32-case 正确性后再计时。


## 最终统一复测

父版本：v16；决定：无稳定收益。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 18.534 | 10.640 | 57.41 |
| 1,1024,8,128 | False | 30.190 | 16.314 | 54.04 |
| 1,4096,8,128 | False | 100.533 | 53.875 | 53.59 |
| 1,8192,8,128 | False | 382.723 | 202.405 | 52.89 |
| 1,4096,8,128 | True | 102.600 | 53.230 | 51.88 |
| 1,8192,8,128 | True | 310.232 | 104.394 | 33.65 |
| 2,4096,16,128 | False | 395.013 | 214.342 | 54.26 |
| 2,4096,16,128 | True | 264.853 | 120.896 | 45.65 |
