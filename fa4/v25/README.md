# v25：5-slot K/V 交替环形 buffer

v24 分离 K/V 生命周期后有约 1-3% 提升。进一步让 K 与 V 交替复用 5 个相同大小的 slots，
替代固定的两个 K slots + 两个 V slots。Load step=2*j 表示 K，2*j+1 表示 V；
slot=step%5，phase=(step//5)%2。QK 释放 K 的 slot，PV 释放 V 的 slot。
这样 V2 可在 K0 的 slot 空闲后预取，不必等旧 V0 被 PV 释放；保留多个 future K/V 的 readiness。

Q 两 tiles + KV 五 tiles 约占 225 KiB SMEM，仍低于 B200 227 KiB dynamic limit。
这是 pipeline depth/lifetime 的一次改动；QK/PV、P/S alias、scalar math、causal remap 保持 v24。
风险是环形 parity 与同一物理 slot 的 K/V 交替角色，需要 tail/多 batch/causal 正确性与 sanitizer。


## 最终统一复测

父版本：v24；决定：无额外收益。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 16.578 | 10.640 | 64.18 |
| 1,1024,8,128 | False | 27.741 | 16.314 | 58.81 |
| 1,4096,8,128 | False | 94.710 | 53.875 | 56.88 |
| 1,8192,8,128 | False | 363.098 | 202.405 | 55.74 |
| 1,4096,8,128 | True | 95.952 | 53.230 | 55.48 |
| 1,8192,8,128 | True | 183.722 | 104.394 | 56.82 |
| 2,4096,16,128 | False | 372.387 | 214.342 | 57.56 |
| 2,4096,16,128 | True | 193.229 | 120.896 | 62.57 |
