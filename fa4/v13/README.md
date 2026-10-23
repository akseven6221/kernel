# v13：完整 tile 的 mask fast path

v12 相比 v7 在 non-causal 有小幅提升，使用 v12 为起点。
唯一优化：先判断整 tile 是否有效，只有 sequence tail 或 causal diagonal 的 tile 才逐元素 mask。
旧版本无论当前 KV tile 是否完全在 causal 过去区域，128 个 scores 都执行 bounds/causal 比较与 select；
長序列绝大多数 tiles 完全有效，可跳过这些指令。Row maximum/exp2/rowsum、TC layout 和调度不变。

仍用 257/511 lengths、causal=128/256/257/511 和大 logits 检查 slow path；
fast path 不能把 causal 最后一轮或越界键当成有效。


## 最终统一复测

父版本：v12；决定：保留。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 18.222 | 10.640 | 58.39 |
| 1,1024,8,128 | False | 29.906 | 16.314 | 54.55 |
| 1,4096,8,128 | False | 101.045 | 53.875 | 53.32 |
| 1,8192,8,128 | False | 386.845 | 202.405 | 52.32 |
| 1,4096,8,128 | True | 101.699 | 53.230 | 52.34 |
| 1,8192,8,128 | True | 307.256 | 104.394 | 33.98 |
| 2,4096,16,128 | False | 399.190 | 214.342 | 53.69 |
| 2,4096,16,128 | True | 263.634 | 120.896 | 45.86 |
