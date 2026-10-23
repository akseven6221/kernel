# v39：N=64 与双 CTA occupancy

父版本 v31 的数值路径，改为 M128/N64 单 Q-stage：97 KiB SMEM、256 TMEM columns、256 threads。
目标是每 SM 同时驻留两个 CTA，用 CTA 间并行填补单 Q-stage 的 QK/softmax/PV 等待。
K/V TMA tile 为 64×128，修改 QK N descriptor、PV K loop、V MN-major descriptor 与 P packing。
保留独立 TMA/MMA roles、双 K/V buffers 和 causal query-major reverse。



## 第二轮最终统一复测

父版本：v31；取舍：回退，资源上限改善但实际 TC 活跃度下降。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 14.794 | 10.731 | 72.54 |
| 1,1024,8,128 | False | 25.163 | 16.304 | 64.79 |
| 1,4096,8,128 | False | 161.464 | 53.210 | 32.95 |
| 1,8192,8,128 | False | 643.013 | 203.664 | 31.67 |
| 1,4096,8,128 | True | 87.627 | 50.741 | 57.91 |
| 1,8192,8,128 | True | 312.677 | 105.736 | 33.82 |
| 2,4096,16,128 | False | 620.360 | 217.224 | 35.02 |
| 2,4096,16,128 | True | 332.027 | 121.982 | 36.74 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "174.016000",
  "launch__registers_per_thread": "128",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "33.142703",
  "sm__warps_active.avg.per_cycle_active": "5.961740",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "0",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "0",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
