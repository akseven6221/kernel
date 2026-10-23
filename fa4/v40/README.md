# v40：两个 Q-stage 交替占用 SFU

父版本 v31。官方源码的 s0_s1_sequence 提示：两个 software WG 同时争用 EX2，
可能延迟首个 P 的完成，导致 Tensor Core 等待。增加两个 128-arrival sequence barriers，
让 stage0(j) exponent → stage1(j) exponent → stage0(j+1)，max reduction 可提前并行。
stage1 的额外 causal 尾 tile 不等待已退出的 stage0；所有迭代都用明确 parity。



## 第二轮最终统一复测

父版本：v31；取舍：回退。
40 项独立 FP32 正确性检查通过，源码 / benchmark / NCU 快照 SHA256 一致。
[全部样本](results.json) · [原始 NCU](profile/profile.ncu-rep) · [NCU 诊断](profile/ncu_details.txt) · [编译与 spill](profile/build.txt)

| B,S,H,D | causal | 本版 us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 15.622 | 10.731 | 68.69 |
| 1,1024,8,128 | False | 24.814 | 16.304 | 65.70 |
| 1,4096,8,128 | False | 81.283 | 53.210 | 65.46 |
| 1,8192,8,128 | False | 310.072 | 203.664 | 65.68 |
| 1,4096,8,128 | True | 82.250 | 50.741 | 61.69 |
| 1,8192,8,128 | True | 158.181 | 105.736 | 66.85 |
| 2,4096,16,128 | False | 331.493 | 217.224 | 65.53 |
| 2,4096,16,128 | True | 174.387 | 121.982 | 69.95 |

NCU 定位数据（4096 non-causal；replay 时间只用于诊断）：

```json
{
  "gpu__time_duration.sum": "92.096000",
  "launch__registers_per_thread": "168",
  "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active": "51.413231",
  "sm__warps_active.avg.per_cycle_active": "9.979884",
  "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "49664",
  "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "24320",
  "derived__memory_l1_wavefronts_shared_excessive": "131072"
}
```
