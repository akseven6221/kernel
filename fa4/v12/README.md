# v12：将 correction / epilogue 合并到 softmax owner

v11 P split 没有稳定收益，继续从 v7 出发。
NCU 的大量 samples 落在 SFULL/CFULL polling，而 conditional rescaling 已使绝大多数 alpha=1。
尝试去掉每轮 softmax -> correction -> MMA 的 mailbox 往返：两个 softmax WGs 各自负责自己 stage 的
罕见 O rescale 和最终 epilogue。CTA 从 512 改成 384 threads，初始 maxnreg=160，
softmax=200，TMA/MMA WG=48；保留两个 Q stages 与双 K/V slots。

这是与章节默认四角色结构不同的优化变体；完整四角色复刻仍在 v7/v11 保留。
Epilogue 使用两个互不重叠的 FP16 shared staging buffers，统一等待最后一个 stage-1 PV 完成后
才复用 Q/K 存储，避免 causal stage0 提早结束时覆盖 stage1 仍要使用的 Q。
PFULL 包含 P stores 和必要 O rescale 的完成，因此去掉 CACK 不会允许 PV 在 correction 前读取 O。

首轮 causal=256 正确性失败，记录在 `runs/v12_merged_correction`，未计入性能结果。
原因是 stage0 提前等待 stage1 尚未进入的未来 parity；parity wait 不能用来等待未推进到的未来 epoch。
修正为独立的 FINAL barrier：MMA 在全部 PV 提交后 commit，两个 softmax groups 等待 phase0 的单次
final completion，随后才复用 Q/K SMEM。避免 stage0 epilogue 在 stage1 还要计算时覆盖 Q1。


## 最终统一复测

父版本：v7；决定：保留。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 20.069 | 10.640 | 53.02 |
| 1,1024,8,128 | False | 33.582 | 16.314 | 48.58 |
| 1,4096,8,128 | False | 116.326 | 53.875 | 46.31 |
| 1,8192,8,128 | False | 446.259 | 202.405 | 45.36 |
| 1,4096,8,128 | True | 132.882 | 53.230 | 40.06 |
| 1,8192,8,128 | True | 400.469 | 104.394 | 26.07 |
| 2,4096,16,128 | False | 456.930 | 214.342 | 46.91 |
| 2,4096,16,128 | True | 335.125 | 120.896 | 36.07 |
