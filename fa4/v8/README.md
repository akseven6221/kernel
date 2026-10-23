# v8：SFU + packed FMA 多项式指数

唯一 softmax 修改：一半元素继续用硬件 `ex2.approx`，另一半使用 packed FP32 FMA 三次多项式。
每两个 score 使用 `f32x2` 的范围缩减/3 次 FMA/指数位重建，避免只依赖 SFU。
多项式系数和范围缩减来自 Dao-AILab `flash_attn/cute/utils.py`，原项目 BSD-3-Clause；
快照和许可见 `../reference/`。保持 v7 的 pipeline 和 barrier 完全相同。

v7 已通过 24 项正确性与 7 项性能检查，8192 non-causal 约 471 us、官方约 202 us。
两组 softmax 同时使用 SFU，先尝试网页中的混合 exponent 路径；后续根据 NCU 决定是否调整比例。
近似值必须重新对照 FP32 reference，尤其是 tail masks、zero 和 large_logits。


## 最终统一复测

父版本：v7；决定：回退，保留实验。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 24.893 | 10.640 | 42.74 |
| 1,1024,8,128 | False | 40.339 | 16.314 | 40.44 |
| 1,4096,8,128 | False | 138.021 | 53.875 | 39.03 |
| 1,8192,8,128 | False | 509.670 | 202.405 | 39.71 |
| 1,4096,8,128 | True | 148.248 | 53.230 | 35.91 |
| 1,8192,8,128 | True | 437.285 | 104.394 | 23.87 |
| 2,4096,16,128 | False | 527.531 | 214.342 | 40.63 |
| 2,4096,16,128 | True | 378.886 | 120.896 | 31.91 |
