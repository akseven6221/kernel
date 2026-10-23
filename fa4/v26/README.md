# v26：教程 non-causal 数据路径整合版

用于完整检查教程结构的整合对照，不把版本号当作“当前最快”。
四个 WGs：WG0/WG1 softmax，WG2 correction/epilogue，WG3 warp0 MMA、warp1 TMA load、warp2 TMA store。
两个 Q stages、TMEM S/P/O 别名、5-slot K/V ring、conditional rescaling、96+32/64+64 P split、
SFU + packed cubic polynomial，以及最终 O 的 swizzled SMEM -> TMA store 都保留。
主体基于已经验证的 v11；复用 v25 的环形生命周期和 v8 的 polynomial，新增独立 output TMA role。

Output tensor map 用 D/H/S/B 四维，避免 flatten B*S 后的尾部 TMA store 跨 batch 写入。
WG2 为两个独立 32 KiB buffers 写 swizzle128 output；proxy fence + EPI_FULL 通知 store warp。
Store warp 等待完整 bulk-group 完成后通知 STORE_DONE，最后 EPIDONE 才允许 TMEM dealloc。
所有 PV 的 FINAL barrier 保护 Q/output SMEM alias。Output map 只在首次绑定 output 时准备，warmup 先于 graph capture。

这是教程重点讲解的 non-causal 路径的 CUDA/PTX 对照；causal 仍用 WG2 epilogue，
与文章 causal specialization 把 epilogue 放回 softmax WGs 的安排不同。范围仍是 dense FP16 forward / D=128。


## 最终统一复测

父版本：v11；决定：教程 non-causal 路径对照。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 23.341 | 10.640 | 45.59 |
| 1,1024,8,128 | False | 38.403 | 16.314 | 42.48 |
| 1,4096,8,128 | False | 133.970 | 53.875 | 40.21 |
| 1,8192,8,128 | False | 499.290 | 202.405 | 40.54 |
| 1,4096,8,128 | True | 145.573 | 53.230 | 36.57 |
| 1,8192,8,128 | True | 434.822 | 104.394 | 24.01 |
| 2,4096,16,128 | False | 512.546 | 214.342 | 41.82 |
| 2,4096,16,128 | True | 377.042 | 120.896 | 32.06 |
