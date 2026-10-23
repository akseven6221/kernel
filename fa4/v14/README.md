# v14：FMA 生成 exponent argument

v13 的 valid-tile mask fast path 改善了长序列，4096 non-causal 从 116.5 降到 100.8 us，
8192 causal 从 401.3 降到 306.3 us。继续减少 softmax 中非矩阵指令。
唯一修改：每 row 预先计算 `-reference*scale_log2`，逐元素的 `(score-reference)*scale_log2`
改为 `fmaf(score,scale_log2,scaled_ref)`，用一个 FFMA 代替 FADD+FMUL。
保持 alpha 的原计算、online reference、所有 barrier 和输出方式。

FMA 与先减再乘的舍入不同；通过 FP32 reference 和大 logits 检查后才能用于性能结论。

最后数值压力检查加入 Q/K=30000 的恒定大 logits，发现 FMA reference rounding 的公共偏移
会导致 FP16 P 溢出（最小复现 `runs/v19_extreme_before_fix`）。最终源码在 `abs(reference)>65536`
时走先减再乘的稳定路径，让当前 maximum 的指数输入精确为 0；普通输入仍用 FMA fast path。
最终统一 comparison / correctness 重测这个修正。


## 最终统一复测

父版本：v13；决定：保留。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 19.144 | 10.640 | 55.58 |
| 1,1024,8,128 | False | 31.654 | 16.314 | 51.54 |
| 1,4096,8,128 | False | 106.994 | 53.875 | 50.35 |
| 1,8192,8,128 | False | 407.381 | 202.405 | 49.68 |
| 1,4096,8,128 | True | 106.656 | 53.230 | 49.91 |
| 1,8192,8,128 | True | 323.158 | 104.394 | 32.30 |
| 2,4096,16,128 | False | 416.874 | 214.342 | 51.42 |
| 2,4096,16,128 | True | 275.886 | 120.896 | 43.82 |
