# v1：串行 Blackwell 数据路径基线

128 query × 128 key tile；单 CTA 128 threads，TMA swizzle128，QK/PV 都使用
`tcgen05.mma.cta_group::1.kind::f16`。S 占 TMEM [0,128)，softmax 将 packed FP16 P 覆盖
前 64 columns；O 占 [128,256)。每 thread 处理一行，采用精确硬件 exp2 的 online softmax，
每次 reference maximum 上升都重缩放 O。输出逐行直接写 global。

保留无流水线版本，作为后续优化的可测基线；此版没有章节中的双 Q stage 和 warp specialization。
最初编译遇到不存在的 `__half2_as_uint` intrinsic，改为 `__half2_raw` 的位打包；失败日志保留。


## 最终统一复测

父版本：无；决定：基线。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 31.840 | 10.640 | 33.42 |
| 1,1024,8,128 | False | 43.989 | 16.314 | 37.09 |
| 1,4096,8,128 | False | 228.646 | 53.875 | 23.56 |
| 1,8192,8,128 | False | 829.216 | 202.405 | 24.41 |
| 1,4096,8,128 | True | 216.688 | 53.230 | 24.57 |
| 1,8192,8,128 | True | 612.062 | 104.394 | 17.06 |
| 2,4096,16,128 | False | 897.048 | 214.342 | 23.89 |
| 2,4096,16,128 | True | 638.728 | 120.896 | 18.93 |
