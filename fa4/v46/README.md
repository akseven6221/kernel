# v46：软件/metadata 预算 240/24

父版本 v44，只调整动态 register 分配。完整 384-thread CTA，每 subpartition 的预算仍在 16384 内。
增加软件寄存器、减少 metadata 配额，测试相反于 v28 的预算方向；以 spill 和配对实测决定。



## 第二轮最终统一复测

父版本：v44；取舍：回退。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 15.122 | 10.731 | 70.97 |
| 1,1024,8,128 | False | 23.405 | 16.304 | 69.66 |
| 1,4096,8,128 | False | 73.654 | 53.210 | 72.24 |
| 1,8192,8,128 | False | 281.810 | 203.664 | 72.27 |
| 1,4096,8,128 | True | 77.861 | 50.741 | 65.17 |
| 1,8192,8,128 | True | 147.011 | 105.736 | 71.92 |
| 2,4096,16,128 | False | 301.379 | 217.224 | 72.08 |
| 2,4096,16,128 | True | 171.210 | 121.982 | 71.25 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "87.424000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "64.002103",
  "sm__warps_active.avg.per_cycle_active": "9.959248",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "68992",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "24192",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
