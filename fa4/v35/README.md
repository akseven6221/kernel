# v35：320 threads，保留原寄存器预算

父版本 v31。v34 的 200 regs/thread 在启动时返回 701；v35 初始设计尚未执行，
已根据该资源约束修正为原来的 entry=168、software=224、metadata=48。
现在唯一修改是删除 WG2 两个未承担工作的 warps。旧初始设计快照保留在失败 run 中。

修正版在首个输入上未完成；独立 90 秒超时复核退出 -9，已保留 build/SASS/源码和执行日志。推测与不完整 metadata warpgroup 执行 setmaxnreg 有关；未将此未验证方案纳入性能排名。
