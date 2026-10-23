# v38：统一先减后乘的指数参数

父版本 v31。FMA fast path + 大 reference fallback 让内层 exponent 代码出现两份。
统一使用 (score-reference)*scale，增加一次普通 FP32 指令，但减少分支、代码体积和寄存器生存期。
这是在新流水线上重新评估 v13 的数值路径，仍覆盖恒定超大 logits。



## 第二轮最终统一复测

父版本：v31；取舍：回退。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 15.146 | 10.731 | 70.85 |
| 1,1024,8,128 | False | 23.390 | 16.304 | 69.70 |
| 1,4096,8,128 | False | 73.960 | 53.210 | 71.94 |
| 1,8192,8,128 | False | 284.266 | 203.664 | 71.65 |
| 1,4096,8,128 | True | 75.894 | 50.741 | 66.86 |
| 1,8192,8,128 | True | 143.243 | 105.736 | 73.82 |
| 2,4096,16,128 | False | 309.843 | 217.224 | 70.11 |
| 2,4096,16,128 | True | 164.136 | 121.982 | 74.32 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "87.648000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "59.024658",
  "sm__warps_active.avg.per_cycle_active": "9.970571",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "28800",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "22784",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
