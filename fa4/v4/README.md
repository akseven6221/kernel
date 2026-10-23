# v4：conditional rescaling

唯一优化是网页的 conditional rescaling：候选 maximum 与当前 reference 的 base-2 差值
超过 8 才更新 reference；其余迭代 alpha=1。每 warp 若所有行 alpha==1，就跳过 O 的
TMEM -> registers -> TMEM 重缩放。rowsum 使用相同 reference 和 alpha，数学更新保持一致。

v3 的合并 stores 改善了 latency，但 8192 non-causal 仍约 684 us、官方约 205 us。
主循环的每个 K/V block 都读写完整 O，单 Q stage 串行 TMEM 路径仍是待减开销。
此版单独检查去除大多数 correction readbacks 的收益；large_logits 测试覆盖强制 rescale。
阈值限制新 P 不超过 2^8，避免延迟更新造成 FP16 exponent 溢出。


## 最终统一复测

父版本：v3；决定：小幅收益。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 19.152 | 10.640 | 55.56 |
| 1,1024,8,128 | False | 29.179 | 16.314 | 55.91 |
| 1,4096,8,128 | False | 176.565 | 53.875 | 30.51 |
| 1,8192,8,128 | False | 688.288 | 202.405 | 29.41 |
| 1,4096,8,128 | True | 159.114 | 53.230 | 33.45 |
| 1,8192,8,128 | True | 462.549 | 104.394 | 22.57 |
| 2,4096,16,128 | False | 721.290 | 214.342 | 29.72 |
| 2,4096,16,128 | True | 467.018 | 120.896 | 25.89 |
