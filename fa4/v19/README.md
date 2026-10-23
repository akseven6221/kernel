# v19：packed FP32 softmax 算术

v17 的 x128 load 与 v18 的单 KV slot 没有收益，回到 v16。
直接检查官方 SASS 时发现大量 `FFMA2`，而自己的 scaling/sum 仍是 scalar 指令。
本版将每两个 score 的 FMA argument generation 和 exponent pair 的 rowsum 累加改为
PTX `fma.rn.ftz.f32x2` / `add.rn.ftz.f32x2`。仍用硬件 scalar exp2，不引入 polynomial；
max reduction、pipeline、layout、register budget 均保持 v16。

两个 sum components 最后合并，加法顺序与旧版不同；必须检查大 logits / causal / tails 的数值误差。
不能把 NCU scalar FP32 instruction counter 的下降直接当作 work 消失，需检查 packed SASS 和 event timing。

最后数值压力检查加入 Q/K=30000 的恒定大 logits，发现 FMA reference rounding 的公共偏移
会导致 FP16 P 溢出（最小复现 `runs/v19_extreme_before_fix`）。最终源码在 `abs(reference)>65536`
时走先减再乘的稳定路径，让当前 maximum 的指数输入精确为 0；普通输入仍用 FMA fast path。
最终统一 comparison / correctness 重测这个修正。


## 最终统一复测

父版本：v16；决定：non-causal 回退。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 17.610 | 10.640 | 60.42 |
| 1,1024,8,128 | False | 28.670 | 16.314 | 56.90 |
| 1,4096,8,128 | False | 96.387 | 53.875 | 55.89 |
| 1,8192,8,128 | False | 365.126 | 202.405 | 55.43 |
| 1,4096,8,128 | True | 95.163 | 53.230 | 55.94 |
| 1,8192,8,128 | True | 286.582 | 104.394 | 36.43 |
| 2,4096,16,128 | False | 377.387 | 214.342 | 56.80 |
| 2,4096,16,128 | True | 246.904 | 120.896 | 48.96 |
