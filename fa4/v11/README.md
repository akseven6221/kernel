# v11：P split completion / PV pipeline

v10 的 independent reduction chains 增加编译器状态和 spill，没有带来加速，回到 v7。
此版补上章节的 P 分段交接：non-causal 96+32、causal 64+64。先计算和写回前一段 P，
用 PFULL 通知 MMA；随后继续计算后半段并以 PTAIL 交接。MMA 可以在后半指数尚未完成时
开始前段 PV，保持 O 的 accumulate 和 KV slot lifecycle 不变。

增加独立 tail full barriers，不能用前段完成代替整个 P tile 的完成。
QK 对 S/P alias 的重用仍等待完整 PV 的 PVDONE，避免 next scores 覆盖正在被 PV 读取的 P。


## 最终统一复测

父版本：v7；决定：无稳定收益。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 23.451 | 10.640 | 45.37 |
| 1,1024,8,128 | False | 37.160 | 16.314 | 43.90 |
| 1,4096,8,128 | False | 121.848 | 53.875 | 44.22 |
| 1,8192,8,128 | False | 462.469 | 202.405 | 43.77 |
| 1,4096,8,128 | True | 138.538 | 53.230 | 38.42 |
| 1,8192,8,128 | True | 408.466 | 104.394 | 25.56 |
| 2,4096,16,128 | False | 477.928 | 214.342 | 44.85 |
| 2,4096,16,128 | True | 355.018 | 120.896 | 34.05 |
