# v7：双 Q stage 与 Warp specialization

把主调度从单 warpgroup 的串行循环改为章节的四角色结构：WG0/WG1 处理两个 Q stages 的 softmax；
WG2 correction/epilogue；WG3 warp0 发 MMA、warp1 发 TMA。两个 Q tiles 共用双 slot K/V，
因此一次 K/V 搬运供两块 Q 使用。S/P/O 的所有交接使用独立 mbarriers。

TMEM 512 columns：S0 [0,128)，S1 [128,256)，O0 [256,384)，O1 [384,512)；
packed FP16 P0/P1 复用 S 的高半区 [64,128)/[192,256)。这与网页给出的布局一致。
WG0/WG1 setmaxnreg=200，WG2=64，WG3=48，动态 register budget 总计 65536。

这是一次架构层面的流水线调整。沿用 v5 的 conditional rescaling、硬件 exp2、一次 reciprocal 和合并输出；
不同于网页默认路径的两点：K/V depth 为 2；output 使用合并 global stores，没有独立 TMA store warp。
此版先不加入 P split completion 或混合 polynomial exponent，以便检查角色分工本身。

风险检查重点：P/S alias 的读写生命周期、KV slot 的 empty parity、最后一轮 alpha/rowsum mailbox 重用、
TMEM deallocation 必须发生在 WG2 全部读出完成之后。Causal 分别记录两个 stages 的 block count。

编译排查已保留 `runs/v7_specialized`、`v7_maxnreg`、`v7_unbounded`：512-thread launch bound /
128-register 上限下，一条 x128 TMEM load 的约束无法分配（要求至少 146）；取消 entry 上限则
setmaxnreg 被忽略，launch 报 701。最终使用显式 `__maxnreg__(128)` 和两条 x64 load，
保留每 thread 的完整 128-score row。ptxas 静态报告 128 registers，动态 softmax SASS 使用 R195，
局部 spill 记录见 build.txt（non-causal 100 B store/136 B load）；没有隐藏这些 compiler costs。


## 最终统一复测

父版本：v5；决定：教程结构对照。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 23.245 | 10.640 | 45.77 |
| 1,1024,8,128 | False | 36.784 | 16.314 | 44.35 |
| 1,4096,8,128 | False | 123.989 | 53.875 | 43.45 |
| 1,8192,8,128 | False | 451.902 | 202.405 | 44.79 |
| 1,4096,8,128 | True | 133.307 | 53.230 | 39.93 |
| 1,8192,8,128 | True | 400.251 | 104.394 | 26.08 |
| 2,4096,16,128 | False | 471.336 | 214.342 | 45.48 |
| 2,4096,16,128 | True | 344.698 | 120.896 | 35.07 |
