# v28：重新分配 MMA/TMA 寄存器预算

父版本 v24，单项预算实验。原 profile 发现 local spill，MMA descriptor 与 barrier 地址的 live range 较长。
软件 WG 从 224 降到 216 regs/thread，元数据 WG 从 48 提升到 64；总角色预算仍为 63488 registers。
检查编译 spill 变化和实测性能，避免只根据 occupancy 推断收益。



## 第二轮最终统一复测

父版本：v24；取舍：回退。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 16.954 | 10.731 | 63.30 |
| 1,1024,8,128 | False | 28.115 | 16.304 | 57.99 |
| 1,4096,8,128 | False | 94.933 | 53.210 | 56.05 |
| 1,8192,8,128 | False | 365.944 | 203.664 | 55.65 |
| 1,4096,8,128 | True | 100.181 | 50.741 | 50.65 |
| 1,8192,8,128 | True | 194.973 | 105.736 | 54.23 |
| 2,4096,16,128 | False | 381.646 | 217.224 | 56.92 |
| 2,4096,16,128 | True | 205.362 | 121.982 | 59.40 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "106.784000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "43.387181",
  "sm__warps_active.avg.per_cycle_active": "9.901735",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "20736",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "22528",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
