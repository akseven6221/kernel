# v16：register budget 调整

v15 的小型 max ILP 在主要长序列场景约有 2-3% 提升。
此版只调整 register 配额：entry 从 160 到 168，softmax 从 200 到 224，TMA/MMA 仍为 48。
总 role budget `256*224 + 128*48 = 63488`，小于 entry pool `384*168 = 64512` 和 SM 的 65536。
依据 v14 NCU 的 local/long-scoreboard 开销与 ptxas 的 spill 记录，检查额外 register 空间能否减少 compiler spills。
必须检查 launch resource 与 setmaxnreg；不能在 pool 不足时请求更多 registers。

最后数值压力检查加入 Q/K=30000 的恒定大 logits，发现 FMA reference rounding 的公共偏移
会导致 FP16 P 溢出（最小复现 `runs/v19_extreme_before_fix`）。最终源码在 `abs(reference)>65536`
时走先减再乘的稳定路径，让当前 maximum 的指数输入精确为 0；普通输入仍用 FMA fast path。
最终统一 comparison / correctness 重测这个修正。


## 最终统一复测

父版本：v15；决定：保留。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 18.536 | 10.640 | 57.40 |
| 1,1024,8,128 | False | 30.197 | 16.314 | 54.02 |
| 1,4096,8,128 | False | 100.350 | 53.875 | 53.69 |
| 1,8192,8,128 | False | 383.254 | 202.405 | 52.81 |
| 1,4096,8,128 | True | 102.440 | 53.230 | 51.96 |
| 1,8192,8,128 | True | 310.190 | 104.394 | 33.65 |
| 2,4096,16,128 | False | 395.730 | 214.342 | 54.16 |
| 2,4096,16,128 | True | 264.835 | 120.896 | 45.65 |
