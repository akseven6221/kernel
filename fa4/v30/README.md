# v30：消除运行时索引的循环边界数组

父版本 v27。SASS 中出现动态地址 LDL，原代码 lengths[stage] 的 stage 来自 threadIdx，
使两个整数的边界数组进入 local memory。改为 len0/len1 和标量选择，保留所有调度与数值操作。
用 ptxas spill / local traffic 与同机 benchmark 检查是否解除长 scoreboard 依赖。



## 第二轮最终统一复测

父版本：v27；取舍：回退，compiler spill 反而增加。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 15.842 | 10.731 | 67.74 |
| 1,1024,8,128 | False | 24.779 | 16.304 | 65.80 |
| 1,4096,8,128 | False | 78.787 | 53.210 | 67.54 |
| 1,8192,8,128 | False | 300.016 | 203.664 | 67.88 |
| 1,4096,8,128 | True | 77.150 | 50.741 | 65.77 |
| 1,8192,8,128 | True | 144.797 | 105.736 | 73.02 |
| 2,4096,16,128 | False | 315.536 | 217.224 | 68.84 |
| 2,4096,16,128 | True | 166.130 | 121.982 | 73.43 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "93.312000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "58.923416",
  "sm__warps_active.avg.per_cycle_active": "9.970517",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "57984",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "22144",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
