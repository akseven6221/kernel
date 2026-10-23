# 同步协议和修正记录

Blackwell TC 的跨 thread producer/consumer 使用完成等待、`tcgen05.fence::before_thread_sync`、
release/acquire 同步及 consumer 的 `tcgen05.fence::after_thread_sync`。
依据 [PTX ISA: TC memory consistency](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#tcgen05-memory-consistency-model)。
最初版本的 ST wait 已有显式完成等待；协议复核后仍补齐 before fence。v1-v8 的旧快照留在历史 runs，
最终 comparison 与全部 26 版 NCU 报告已覆盖修改后的源码，并核对 SHA256。

v7/v11 的 slot 所有权：

| 资源 | producer -> consumer | 交接/释放 |
| --- | --- | --- |
| Q SMEM | TMA -> MMA | QFULL，整 kernel 保留 |
| K/V slot | TMA -> MMA -> TMA | KVFULL，全部 PV commit 后 KVEMPTY |
| S TMEM | QK -> softmax | SFULL；下一 QK 先等待上次 PVDONE，避免覆盖别名 P |
| P TMEM | softmax -> PV | ST wait + before fence + PFULL；v11 另有 PTAIL |
| O TMEM | PV -> correction -> PV | PVDONE + CACK |
| alpha/sum SMEM | softmax -> correction | CFULL + CACK；最终 rowsum 写入前等待最后 CACK |
| TMEM allocation | WG2 epilogue -> allocator warp | LD wait + before fence + EPIDONE |

v12 的 correction 与 softmax 合并，PFULL 同时交接 P 和已经 correction 的 O。
它曾在 causal stage0 比 stage1 早一轮结束时错误等待 stage1 的未来 parity；当前 phase 尚未推进时，
`try_wait.parity` 可能把该 future parity 判断为已完成。改为一次性的 FINAL barrier：
MMA 提交所有 PV 后 commit FINAL，两个 software owners 在 phase0 等到真正 final completion，
才复用 Q/K 的 SMEM。失败输入与源码见 `runs/v12_merged_correction`，通过的修正版见 `runs/v12_final_barrier`。

Sanitizer 的小型实测覆盖不证明任意数据/任何 schedule 都正确。最终审计增加 causal/non-causal 的 S=511，
覆盖 KV slot reuse、partial tiles 和不同 Q-stage 的退出时间。工具输出及 exit codes 作为证据保留。

v24 将 K 与 V 的 full/empty barrier 分开：QK 等待 KFULL，两个 Q-stage 的 QK 使用结束后释放 KEMPTY；
PV 等待 VFULL，两个 Q-stage 的 PV 使用结束后释放 VEMPTY。这样下次 K TMA 无需等待当前 V 的全部 PV。
v25/v26 改为交替 5-slot ring：第 j 轮 K/V 的 step 为 2j/2j+1，slot=step%5，phase=(step/5)&1，
每个 slot 必须由相应 QK/PV 最后一次使用的 completion 释放后才能重新加载。

v26 的 output 复用 Q SMEM 前，仍等待独立 FINAL，防止 MMA 尚在读取 Q。
WG2 correction 将 FP16 output 写入 swizzle-128 SMEM，执行 `fence.proxy.async.shared::cta` 后到达 EPI_FULL；
独立 output warp 等待 EPI_FULL，再用 rank-4 TMA map（D/H/S/B）写出两个 D=64 chunk，避免 partial tile 跨 batch。
每个 Q-stage 的 bulk commit / wait_group 0 后发出 STORE_DONE；WG2 等待两个 STORE_DONE 后才到达 EPIDONE，
允许 allocator warp 释放 TMEM。最终 v24/v25/v26 各 6 项 sanitizer 均零错误、零警告，原始输出位于
`runs/final_sanitizers/`。

## 第二轮 v27–v51

v27 改为 QK0(j)、PV1(j−1)、QK1(j)、PV0(j) 的交错 issue 顺序；K/V 的释放仍由最后一次
读取该 slot 的 MMA completion 控制，末尾单独 drain stage1 的 PV。

v31 移除 MMA issuer 内用于 P/S 别名复用的 PVDONE 等待和 commit。依据
[PTX ISA 的 pipelined tcgen05 顺序规则](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-pipelined-instructions)，
同一 warp、相同 cta_group 发出的 MMA，前一条 A/metadata 读取先于后一条 D 写入。
因此 PV 读取 P 后，后续 QK 才能覆盖同一块 TMEM。此规则不替代跨线程同步：
softmax 的 TMEM store 仍等待完成并执行 before fence，然后发出 PFULL；issuer 等待 PFULL 后执行 after fence。
SFULL 的 commit 仍覆盖此前发出的 MMA，使 software owner 在 correction O 前确认前次 PV 完成；
FINAL 仍保护末尾 epilogue 的共享内存复用和 TMEM 释放。

v31 的六项 sanitizer、96 项额外正确性检查保存在 `runs/r2_v31_sanitizer/` 和 `runs/r2_v31_stress/`。
v34 的 320-thread register 配置返回启动错误 701；v35 的动态预算版本在首个输入上停滞并被超时终止。
这两版未通过验证，不能作为有效同步实现；v35 的具体硬件停滞原因尚未证实。

v51 包含 v44 双 Q-stage 与 v50 单 Q-stage，启动时只选择一个完整 kernel。
两条路径均保留各自的 slot completion、PFULL、SFULL 和 FINAL 协议。
`runs/r2_v51_sanitize_small/`（S=511）和 `runs/r2_v51_sanitize_large/`（S=8191）分别覆盖两条路径，
各包含两种 causal 模式的 memcheck/synccheck/racecheck，共 12 项退出 0、零错误。
120 项额外 FP32 压力检查保存在 `runs/r2_v51_stress/`；上述检查不构成对任意 schedule 的形式化证明。
