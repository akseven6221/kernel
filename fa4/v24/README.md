# v24：分离 K/V readiness 与 lifetime

v23 仍让 QK 等 K 和 V 两个 TMA tiles 都完成，而且 K slot 一直占用到 PV 完成。
但 QK 只需要 K；两个 QK 完成后 K 的 buffer 可以先重用，V 则应保留到 PV 完成。
此版把 KV full/empty 各拆成独立 K/V barriers：QK 等 KFULL，QK commits KEMPTY；
PV 前等 VFULL，PV commits VEMPTY。保持两个 K slots 和两个 V slots 的实际 SMEM 布局不变。
Loader 可在旧 V 尚在被 PV 使用时预取下一轮的 K，去掉不必要的 K->V 连带等待。

这是流水线 lifetime 的单项改动；P/S alias、O correction、FINAL/EPIDONE 等保护保持原样。
需重点检查 TMA overwrite 与 QK/PV 的异步读是否正确交接。


## 最终统一复测

父版本：v23；决定：主要推荐。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 16.784 | 10.640 | 63.39 |
| 1,1024,8,128 | False | 27.776 | 16.314 | 58.73 |
| 1,4096,8,128 | False | 94.246 | 53.875 | 57.16 |
| 1,8192,8,128 | False | 360.952 | 202.405 | 56.08 |
| 1,4096,8,128 | True | 94.755 | 53.230 | 56.18 |
| 1,8192,8,128 | True | 181.642 | 104.394 | 57.47 |
| 2,4096,16,128 | False | 370.965 | 214.342 | 57.78 |
| 2,4096,16,128 | True | 191.168 | 120.896 | 63.24 |
