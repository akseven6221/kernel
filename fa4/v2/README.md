# v2：直接硬件 exp2

唯一性能修改：将通用 CUDA math `exp2f` 改为 PTX `ex2.approx.ftz.f32`。
依据 v1 的 Nsight Compute：SM 吞吐 21.96%、issue slots 26.15%、非融合 FP32 指令 25,161,728；
SASS 中每次 exponent 包含额外的范围判断和特殊值处理。Attention 的正常 softmax 指数无需这些通用分支，
直接硬件 exponent 是网页对应路径。保留 v1 的 tiling、同步、重缩放规则和输出方式，单独测这项变化。

这项修改使用近似指令；需要重新通过普通、零、大 logits 以及 causal/tail 的 FP32 参考检查。


## 最终统一复测

父版本：v1；决定：保留。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 30.411 | 10.640 | 34.99 |
| 1,1024,8,128 | False | 41.563 | 16.314 | 39.25 |
| 1,4096,8,128 | False | 210.157 | 53.875 | 25.64 |
| 1,8192,8,128 | False | 760.104 | 202.405 | 26.63 |
| 1,4096,8,128 | True | 190.650 | 53.230 | 27.92 |
| 1,8192,8,128 | True | 525.330 | 104.394 | 19.87 |
| 2,4096,16,128 | False | 831.718 | 214.342 | 25.77 |
| 2,4096,16,128 | True | 566.154 | 120.896 | 21.35 |
