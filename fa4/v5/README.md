# v5：一次 reciprocal 完成归一化

唯一修改：每 row 计算一次 `rcp.approx.ftz.f32(rowsum)`，128 output 元素改用 multiply。
依据 v3/v4 数据路径的 PTX：epilogue 出现 128 次 `div.rn.f32`；v3 SASS 两个 specialization
合计 260 个 `MUFU.RCP` 和 262 个 `CALL.REL.NOINC`。通用精确除法保留大量特殊值处理，
明显超过每 row 一次正数 reciprocal 的需要。保持所有 tiling、pipeline、softmax 和输出布局不变。


## 最终统一复测

父版本：v4；决定：保留。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 15.669 | 10.640 | 67.91 |
| 1,1024,8,128 | False | 25.949 | 16.314 | 62.87 |
| 1,4096,8,128 | False | 172.445 | 53.875 | 31.24 |
| 1,8192,8,128 | False | 680.171 | 202.405 | 29.76 |
| 1,4096,8,128 | True | 148.920 | 53.230 | 35.74 |
| 1,8192,8,128 | True | 445.970 | 104.394 | 23.41 |
| 2,4096,16,128 | False | 709.726 | 214.342 | 30.20 |
| 2,4096,16,128 | True | 436.405 | 120.896 | 27.70 |
