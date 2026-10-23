# v9：warp 级 barrier arrivals

v8 的 50% polynomial exponent 在 paired benchmark 里更慢，保留为失败性能实验；回到 v7 作起点。
v7 NCU long-scoreboard samples=3930、selected=860，math-pipe-throttle=89，表明当前瓶颈并非 SFU 饱和。
尝试减少同步操作：PFULL/CFULL/CACK/EPIDONE 的 arrival 从每 thread 一次改为每 warp 的 lane0 一次，
计数从 128 改为 4。`__syncwarp()` 保证同 warp 的写入及 TC 完成等待先于 lane0 release。
不改变 owner、phase、softmax 数学、TMA/MMA issue 顺序。

同步协议复核补齐 `tcgen05.fence::before_thread_sync`（PTX ISA 的跨 thread handoff 要求），
该 correctness fix 同步应用到 v1-v8；历史源码完整快照仍在 runs 中，最终统一 benchmark 会重测最终源码。


## 最终统一复测

父版本：v7；决定：无稳定收益。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 23.243 | 10.640 | 45.78 |
| 1,1024,8,128 | False | 36.664 | 16.314 | 44.49 |
| 1,4096,8,128 | False | 120.605 | 53.875 | 44.67 |
| 1,8192,8,128 | False | 450.947 | 202.405 | 44.88 |
| 1,4096,8,128 | True | 135.750 | 53.230 | 39.21 |
| 1,8192,8,128 | True | 408.314 | 104.394 | 25.57 |
| 2,4096,16,128 | False | 469.429 | 214.342 | 45.66 |
| 2,4096,16,128 | True | 350.573 | 120.896 | 34.49 |
