# v6：双 K/V slot 预取

唯一主循环调整：K/V 双 buffer，在计算 tile j 前发起 tile j+1 的 TMA，尝试把搬运与 QK/softmax/PV 重叠。
上个 tile 的 PV completion 和 CTA barrier 保证 slot 在重用前已经空闲；每个 full barrier 使用独立 parity。
Q、softmax、PV 的分工仍是 v5 的串行 128-thread CTA。

v5 的 reciprocal 减少 epilogue 时间，但 8192 non-causal 仍约 670 us，官方约 201 us；
v3 profile 的 long-scoreboard samples=2741，selected=1882，Tensor Core elapsed active 仅 14.9%。
尝试消除每轮 TMA 等待。代价：SMEM 从 96 KiB 增为 160 KiB，会降低 resident CTAs；
因此必须实测后决定是否保留为后续起点，不能仅因使用了异步预取就宣称更快。


## 最终统一复测

父版本：v5；决定：短序列候选。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 14.642 | 10.640 | 72.67 |
| 1,1024,8,128 | False | 23.653 | 16.314 | 68.97 |
| 1,4096,8,128 | False | 151.837 | 53.875 | 35.48 |
| 1,8192,8,128 | False | 580.058 | 202.405 | 34.89 |
| 1,4096,8,128 | True | 130.267 | 53.230 | 40.86 |
| 1,8192,8,128 | True | 379.704 | 104.394 | 27.49 |
| 2,4096,16,128 | False | 533.043 | 214.342 | 40.21 |
| 2,4096,16,128 | True | 361.003 | 120.896 | 33.49 |
