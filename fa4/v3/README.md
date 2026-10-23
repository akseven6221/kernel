# v3：合并 output stores

只改 epilogue：用已结束生命周期的 Q/K/V SMEM 作 128×129 FP32 staging transpose。
Softmax thread 仍负责一整行；结果先按 column-major padding 写 SMEM，随后按连续 head_dim 的 half2 写 GMEM。
不增加 shared memory allocation，不改主循环。Padding 避免转置带来的严重 bank conflicts。

依据 v1 profile（v2 未改 epilogue）：4,194,304 global-store sectors，3,932,160 excessive sectors，
每 sector 仅使用 2/32 bytes。单独验证合并写入是否降低 sectors，以及短/长序列 latency 是否改善。


## 最终统一复测

父版本：v2；决定：保留。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 19.250 | 10.640 | 55.27 |
| 1,1024,8,128 | False | 29.478 | 16.314 | 55.34 |
| 1,4096,8,128 | False | 178.472 | 53.875 | 30.19 |
| 1,8192,8,128 | False | 693.446 | 202.405 | 29.19 |
| 1,4096,8,128 | True | 165.568 | 53.230 | 32.15 |
| 1,8192,8,128 | True | 480.941 | 104.394 | 21.71 |
| 2,4096,16,128 | False | 724.619 | 214.342 | 29.58 |
| 2,4096,16,128 | True | 483.171 | 120.896 | 25.02 |
