# v27：重新安排跨 Q-stage 的 MMA 顺序

父版本 v24。Profile 的 TC active 只有约 47%，两个 QK 后集中等待两个 softmax，
在两个 Q-stage 间仍有流水线空隙。改为 QK0(j)、PV1(j-1)、QK1(j)、PV0(j)，
让一个 stage 的 softmax 与另一个 stage 的 QK/PV 交错。
K 在两个 QK 后释放，V 延后到下一轮 PV1 后释放；FINAL 等待最后一次 PV1。
这是调度实验，性能与正确性以保存的运行结果为准。



## 第二轮最终统一复测

父版本：v24；取舍：保留，长序列首个主要收益。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 15.566 | 10.731 | 68.94 |
| 1,1024,8,128 | False | 24.118 | 16.304 | 67.60 |
| 1,4096,8,128 | False | 77.510 | 53.210 | 68.65 |
| 1,8192,8,128 | False | 291.579 | 203.664 | 69.85 |
| 1,4096,8,128 | True | 77.360 | 50.741 | 65.59 |
| 1,8192,8,128 | True | 147.347 | 105.736 | 71.76 |
| 2,4096,16,128 | False | 308.787 | 217.224 | 70.35 |
| 2,4096,16,128 | True | 167.157 | 121.982 | 72.97 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "90.848000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "56.758554",
  "sm__warps_active.avg.per_cycle_active": "9.971897",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "28800",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "22784",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
