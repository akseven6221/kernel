# v33：低比例 polynomial exponent

父版本 v31。此前 v8/v26 的 50% 混合比例增加了过多 ALU 指令；重新尝试 12.5%，
只在中间两个 32-score fragments 的四分之一 pair 使用 cubic emulation，其余硬件 EX2。
首尾 fragments 保留硬件路径，降低主链延迟。极大 reference 稳定分支保持硬件 EX2。
多项式 helper 来自 v8，原始来源和 BSD-3-Clause 许可仍保存在 reference/。



## 第二轮最终统一复测

父版本：v31；取舍：回退。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 15.403 | 10.731 | 69.67 |
| 1,1024,8,128 | False | 23.846 | 16.304 | 68.37 |
| 1,4096,8,128 | False | 76.216 | 53.210 | 69.81 |
| 1,8192,8,128 | False | 286.376 | 203.664 | 71.12 |
| 1,4096,8,128 | True | 78.195 | 50.741 | 64.89 |
| 1,8192,8,128 | True | 148.467 | 105.736 | 71.22 |
| 2,4096,16,128 | False | 309.035 | 217.224 | 70.29 |
| 2,4096,16,128 | True | 167.690 | 121.982 | 72.74 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "86.464000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "56.626057",
  "sm__warps_active.avg.per_cycle_active": "9.952434",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "28800",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "22784",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
