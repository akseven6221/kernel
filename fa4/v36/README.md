# v36：两个自主工作组

父版本 v31 的数学与布局，重写角色分工：256 threads，各 128-thread WG 负责一个 Q-stage 的
QK→softmax/correction→PV；WG0 leader 兼任双 KV buffer 的 TMA 预取。
取消独立 metadata WG，统一 248 regs/thread；每个 KVEMPTY barrier 等两个 stage 的 commit。
PFULL 由 WG 内 barrier 替代，FINAL 等两个 issuer 的最终 commit，再允许复用 Q SMEM。
重点比较同步开销和寄存器 spill，必须审计两个 issuer 的 KV 生命周期。



## 第二轮最终统一复测

父版本：v31；取舍：回退，零 spill 也没有加速。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 15.206 | 10.731 | 70.57 |
| 1,1024,8,128 | False | 24.882 | 16.304 | 65.53 |
| 1,4096,8,128 | False | 87.802 | 53.210 | 60.60 |
| 1,8192,8,128 | False | 335.154 | 203.664 | 60.77 |
| 1,4096,8,128 | True | 94.221 | 50.741 | 53.85 |
| 1,8192,8,128 | True | 187.246 | 105.736 | 56.47 |
| 2,4096,16,128 | False | 354.198 | 217.224 | 61.33 |
| 2,4096,16,128 | True | 182.419 | 121.982 | 66.87 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "97.792000",
  "launch__registers_per_thread": "235",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "46.989849",
  "sm__warps_active.avg.per_cycle_active": "7.965459",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "0",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "0",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
