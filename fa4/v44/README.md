# v44：按 causal 专门化 softmax

父版本 v42。实测 packed 组合对 non-causal 稳定快约 2%，causal 却稍慢。
编译期让 non-causal 使用 v42 的 packed/max3，causal 回到 v31 的标量参数与四条 sum chains。
数学定义一致；这是基于模式的固定编译选择，不依赖输入数据或计时状态。



## 第二轮最终统一复测

父版本：v42；取舍：作为最终大 grid 实现。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 14.725 | 10.731 | 72.88 |
| 1,1024,8,128 | False | 22.861 | 16.304 | 71.32 |
| 1,4096,8,128 | False | 73.122 | 53.210 | 72.77 |
| 1,8192,8,128 | False | 275.558 | 203.664 | 73.91 |
| 1,4096,8,128 | True | 74.706 | 50.741 | 67.92 |
| 1,8192,8,128 | True | 141.886 | 105.736 | 74.52 |
| 2,4096,16,128 | False | 295.139 | 217.224 | 73.60 |
| 2,4096,16,128 | True | 166.219 | 121.982 | 73.39 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "85.568000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "57.326817",
  "sm__warps_active.avg.per_cycle_active": "9.970988",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "28800",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "22784",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
