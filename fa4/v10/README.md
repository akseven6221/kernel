# v10：缩短逐 row reduction 的依赖链

v9 的 warp arrival 改动没有稳定收益，回到 v7 作起点。
唯一算法内优化是 row reduction 的 scheduling：128-score max 与 64-pair sum 各分为 8 条独立链，
最后用二叉树合并。原代码的 max/psum 依次依赖前一个 FMAX/FADD，限制 softmax 的指令级并行；
此版保留同样的运算量和每 thread 处理一行的布局，仅缩短依赖路径。

psum 的加法顺序改变，会有 FP32 roundoff 差别；再次验证普通、零、大 logits 与 masked tails。
保持纯硬件 exp2，不叠加 v8 的 polynomial 或 v9 的 arrival 修改。


## 最终统一复测

父版本：v7；决定：回退。
当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。
完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。

| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |
| --- | --- | ---: | ---: | ---: |
| 1,512,8,128 | False | 23.875 | 10.640 | 44.57 |
| 1,1024,8,128 | False | 38.179 | 16.314 | 42.73 |
| 1,4096,8,128 | False | 125.646 | 53.875 | 42.88 |
| 1,8192,8,128 | False | 472.874 | 202.405 | 42.80 |
| 1,4096,8,128 | True | 148.045 | 53.230 | 35.96 |
| 1,8192,8,128 | True | 436.549 | 104.394 | 23.91 |
| 2,4096,16,128 | False | 490.869 | 214.342 | 43.67 |
| 2,4096,16,128 | True | 376.582 | 120.896 | 32.10 |
