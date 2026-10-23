# v42：完整 packed FP32 softmax 与四条累加链

父版本 v31。v19 只有一条 packed sum 依赖链，v29 使用标量 rowsum，未同时减少指令和拉开依赖。
此版组合 packed FMA、max3、四条独立 packed sum chains（8 个标量 accumulator），
降低普通 FP32 指令数，同时避免 64 次串行 add 的长依赖。保留大 reference 稳定路径。
这是组合实验，若有收益再拆分消融验证，不能直接把收益归给其中某一项。

首轮生成代码误用了 helper 名称 scale_pair（实际为 argument2），编译失败快照保留在 runs/r2_v42_packed_softmax；已修正后重新运行。



## 第二轮最终统一复测

父版本：v31；取舍：保留 non-causal 路径。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 14.693 | 10.731 | 73.04 |
| 1,1024,8,128 | False | 22.806 | 16.304 | 71.49 |
| 1,4096,8,128 | False | 72.360 | 53.210 | 73.53 |
| 1,8192,8,128 | False | 274.299 | 203.664 | 74.25 |
| 1,4096,8,128 | True | 75.264 | 50.741 | 67.42 |
| 1,8192,8,128 | True | 142.440 | 105.736 | 74.23 |
| 2,4096,16,128 | False | 293.634 | 217.224 | 73.98 |
| 2,4096,16,128 | True | 164.243 | 121.982 | 74.27 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "84.672000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "57.369734",
  "sm__warps_active.avg.per_cycle_active": "9.970718",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "28800",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "22784",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
