# v47：max3 单项消融

父版本 v31，只替换 max reduction 为 max3，保留标量 FMA 和四条 sum chains。
用于判断 v42 组合的收益是否主要来自 max3，避免把 packed arithmetic 的收益误归因。



## 第二轮最终统一复测

父版本：v31；取舍：基本持平，v42 的收益不能单独归给 max3。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 15.264 | 10.731 | 70.30 |
| 1,1024,8,128 | False | 23.509 | 16.304 | 69.35 |
| 1,4096,8,128 | False | 73.826 | 53.210 | 72.07 |
| 1,8192,8,128 | False | 282.150 | 203.664 | 72.18 |
| 1,4096,8,128 | True | 74.829 | 50.741 | 67.81 |
| 1,8192,8,128 | True | 142.554 | 105.736 | 74.17 |
| 2,4096,16,128 | False | 303.214 | 217.224 | 71.64 |
| 2,4096,16,128 | True | 167.787 | 121.982 | 72.70 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "85.216000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "57.056975",
  "sm__warps_active.avg.per_cycle_active": "9.957393",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "28800",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "22784",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
