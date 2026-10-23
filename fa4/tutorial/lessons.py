"""Hand-written teaching notes. Lesson numbers are distinct from experiment IDs."""

LESSONS = []


def lesson(original, title, problem, change, evidence, invariant, exercise, answer, anchors):
    LESSONS.append(dict(original=f'v{original}', title=title, problem=problem,
                        change=change, evidence=evidence, invariant=invariant,
                        exercise=exercise, answer=answer, anchors=anchors))


lesson(1, '跑通串行 attention：先理解数据流',
       '先建立可解释的低性能基线。一块 CTA 处理 128 个 query，每次读 128 个 key/value；搬运、QK、softmax、PV 顺序执行，等待期间缺少另一份工作填空。',
       'Q/K/V 经 TMA 进入 swizzled SMEM；QK 把 FP32 scores 写入 TMEM；每线程取一行，在寄存器中做 online softmax，再把两个 FP16 P 打包成一个 32-bit word 写回 TMEM；PV 累加 FP32 O，最后归一化写出。原版 v1 使用 exp2f，v2 才直接用近似 PTX 指令。',
       'NCU 和 SASS 暴露通用指数、非合并输出及逐元素除法开销。先认识这些可独立处理的问题，再改流水线；v1 已经使用 Tensor Core，低性能不等于没有使用矩阵硬件。',
       'P 覆盖 S 前必须读完 scores；QK 覆盖下一轮 S 前必须结束前次 PV 对 P 的读取。TMEM 分配 256 columns，S/P 位于前 128 列，O 位于后 128 列。',
       '手画一个 CTA 的第 0、1 个 KV tile：rowmax、rowsum、O 各保存什么？为什么不能分别 softmax 每个 tile 后直接相加？',
       '每个 tile 的分母不同。必须统一 reference，并用 alpha 同时重缩放旧 rowsum 和旧 O；最后仅除一次全局 rowsum。',
       [('void qk(', '先读 QK/PV helper：两次 GEMM 的输入位置不同'),
        ('float rowmax=-INFINITY', '每个 query row 持有跨 KV tiles 的 online 状态'),
        ('rowsum=rowsum*alpha+psum;', '分母与 O 必须使用相同的 reference 更新'),
        ('struct Context', 'Host 只负责 tensor map 和 launch，主要学习 kernel 内的数据流')])

lesson(2, '指数指令：exp2f → ex2.approx',
       '每行每 tile 有 128 个 exponent；通用 math 路径的额外指令会被重复很多次。',
       '只替换指数 helper 及其调用。scale_log2=log2(e)/sqrt(128)，因此 exp(score/sqrt(D)) 可以用 exp2(score*scale_log2) 表示。PTX 版本是近似且带 ftz 的数值路径，不能称作精确等价。',
       '对照原 v1 的 SASS 中 exponent 的范围处理，再看 v2 的 MUFU.EX2 和配对 latency。下表用最终修正后源码的统一复测，而非早期试跑数字。',
       'alpha 和 P 的 exponent 都要用同一缩放定义；保留 mask 的 -infinity。容差验证包括尾块、大 logits 和恒定大 logits。',
       '为什么这里比较 allclose，而不是要求 FP16 输出 bitwise 一样？',
       '指数近似及后续 FP16 P 的舍入会改变结果；是否可接受由独立 FP32 参考、有限性和既定容差共同判断。',
       [('float fast_exp2(', '本课唯一核心 helper：近似 base-2 exponent'),
        ('float alpha=', 'reference rescale 也调用同一个 helper')])

lesson(3, '合并写回：用 SMEM 转换线程到输出的映射',
       '一个线程负责一整行适合 softmax，但同一时刻相邻 lanes 写不同 query 行，global 地址跨越 H×D。算得顺手的布局不一定写得高效。',
       '结束主循环后复用已死亡的 Q/K/V SMEM；先写 staging[c*129+row]，同步，再按连续 head_dim 读取两个元素，用 half2 合并写 global。129 是 FP32 staging 的 pitch。',
       '基线输出有 4194304 global-store sectors，其中 3932160 excessive sectors。检查 v3 的 sector 指标和短序列 latency，解释为何固定 epilogue 开销在小任务中更显眼。',
       '只有最后一次 PV 完成后才能复用 SMEM；转置写入与读取之间保留 CTA barrier；最后一个 query tile 仍按 seq 屏蔽输出。',
       '固定 c，比较旧输出地址与新 half2 地址：lane 增加 1 时，地址各增加多少？',
       '旧地址通常跨 H*128 个 half；新映射同一行内相邻 lanes 各写两个 half，形成连续地址。跨行处需结合循环的 i/64 计算。',
       [('float* staging=', 'SMEM 作为线程间交换缓冲，而不是额外 global transpose'),
        ('for(int i=tid;i<128*64', '输出 ownership 从每线程一行改为连续 half2')])

lesson(4, '减少 O 重缩放：延迟更新 reference',
       '标准 online softmax 每当最大值上升就重缩放 O；O 有 128 个 FP32 元素，TMEM 读出、乘 alpha、写回有固定成本。',
       'candidate 与 reference 的 base-2 距离超过 8 才更换 reference，否则 alpha=1；一个 warp 内所有行都无需 rescale 时，跳过该 warp 的 O 读写。此时 rowmax 更准确的名字是 reference，它可能小于实际最大 score。',
       '读主循环中的 __any_sync 和 TMEM correction；同机结果只有较小收益，说明此项不能独自解决串行等待。这里是减少工作量，不是改变 tile。',
       'rowsum 和 O 必须同时以同一个 reference 表示。延迟更新期间最大 exponent 输入不超过 8，P≤256；更大变化及时 rescale。首轮显式初始化，避免 -infinity 运算污染状态。',
       'alpha=1 时为什么仍要更新 rowsum？为什么不能简单让所有 alpha 永远为 1？',
       '新 tile 仍贡献 psum。永不更新 reference 会让指数范围失控，FP16 P 可能溢出；阈值限制了这个范围。',
       [('float candidate=', 'reference 不再每轮追随最大值'),
        ('if(j>0 && __any_sync', 'warp 内任何一行需 correction 才执行集体 TMEM 读写')])

lesson(5, '归一化：一行一次 reciprocal',
       'epilogue 里 128 个 output 元素除以同一个 rowsum，重复精确除法不必要。',
       '每行先算一次 rcp.approx.ftz.f32(rowsum)，随后 o[c]*inverse。主循环、布局、reference 规则均继承前版。',
       '查看 PTX 的 div.rn 和 SASS 的 reciprocal/辅助调用是否减少，再比较短序列。源代码的 / 不一定只对应一条便宜机器指令。',
       '这里仍采用近似数学；有效 query 行的 rowsum 为正。不要把越界行的中间数值当作有效输出。',
       '为什么这项优化更可能改变短序列的相对速度？',
       '归一化成本按 query 行数增长，长序列主循环还按 key 长度增长；固定 epilogue 在短序列占比更大。',
       [('float inverse; asm(', '每行只做一次 reciprocal'),
        ('staging[c*129+tid]=o[c]*inverse;', '把结果广播给本行全部输出元素')])

lesson(6, '预取支线：双 K/V buffer',
       '串行基线每轮先等 TMA，再算 QK/softmax/PV；可以尝试在算 tile j 时加载 j+1。',
       '增加第二组 K/V slot，以 j%2 选择 slot，以 (j/2)&1 追踪同一 slot 的 barrier phase。先填第一个 slot，进入循环后预取下一块。此课是 v5 的支线；下一课也从 v5 出发。',
       'v6 的短序列和长序列都比 v5 快，但双 buffer 让数据 SMEM 从 96 KiB 增到 160 KiB。后续双 Q-stage 的资源与角色分工需重新设计，不能直接把 v6 的改动当成 v7 的父实现。',
       '旧 PV 完成后才能覆盖其 V；旧 QK 完成后才能覆盖 K。轮次 j 的 parity 与 slot 的重用次数相关，不能对所有 slot 使用同一个随意翻转的 phase。',
       '列出 j=0..5 的 slot 和 phase，并标出每个 slot 何时可重新写入。',
       '(slot,phase) 为 (0,0),(1,0),(0,1),(1,1),(0,0),(1,0)。可重用时间由前次消费者完成决定。',
       [('const uint32_t tbar=base+5*TILE_BYTES', '多出的两个 TILE_BYTES 对应第二组 K/V'),
        ('if(tid==0 && j+1<blocks)', '在本轮计算开始前发起下一轮 TMA')])

lesson(7, '架构过渡：双 Q-stage 与四个 warpgroup',
       '单 query tile 的 softmax 和同 tile 的 QK/PV 有数据依赖；仅增加 KV 预取无法让两者自由重叠。',
       '一个 CTA 同时拥有两个 128-row Q tiles。WG0/WG1 分别做 softmax，WG2 做 correction/epilogue，WG3 内分 MMA 与 TMA issuer；一次 K/V 加载供两个 Q stages 使用。512 threads、512 TMEM columns，显式 mbarrier 交接 ownership。这是一次结构重写，不是单条指令优化。',
       '长序列下降明显，短序列反而慢于 v6。四角色建立并行机会，也增加 mailbox、资源和启动开销；这为下一课合并角色提供可观察的起点。',
       'S0/S1 位于 TMEM [0,128)/[128,256)，O0/O1 位于 [256,384)/[384,512)；P 分别覆盖各 S 的高 64 columns。明确 PFULL、PVDONE、CFULL、CACK 各保护谁。',
       '如果 stage0 softmax 已结束，为什么不能立刻释放整块 TMEM？',
       'stage1 或 correction owner 可能仍读写其片段。释放必须等待全部使用者完成，而不是某个线程退出局部循环。',
       [('constexpr int QFULL=', '把 barrier 表与 ownership 图一起看'),
        ('if(wg==3)', 'metadata WG：独立 TMA 与 MMA issuer'),
        ('} else if(wg<2)', '两个 software owners 独立消费各自 SFULL'),
        ('scales[stage*128+local]=alpha;', '跨 WG correction mailbox 是下一课要减少的往返')])

lesson(12, '合并角色：softmax owner 自己修正 O',
       'conditional rescale 让 correction 的实际工作变少，但每轮仍为独立 WG 交换 alpha 和 ack。NCU polling samples 提示通信可能比工作本身更贵。',
       '把 O correction 和 epilogue 放回各自 softmax WG，CTA 从 512 降到 384 threads；PFULL 同时表示 P 与必要 O correction 完成。初始 register budget 和 staging 随角色重构一起调整，不能把性能变化归因于线程数一个变量。',
       'non-causal 小幅改善，causal 并非所有 case 都改善。第一次实现暴露 future parity 错误，修正过程是本课重点：用一次性的 FINAL 等待全体 PV drain。',
       'causal 下 stage0 可能先结束；它复用 Q/K SMEM 作输出前，stage1 必须真正结束。不能用等待未来 parity 代替未来 epoch 的完成事件。',
       '画出 stage0 先退出、stage1 仍在读 Q1 的时间线，解释错误的提前 epilogue 会覆盖什么。',
       'stage0 staging 会复用仍被 stage1 QK 读取的共享内存。FINAL 在所有 PV 提交之后 commit，两个 owners 等其完成后再复用。',
       [('__global__ __maxnreg__(160)', '角色合并后使用 384-thread CTA 与新的 entry register 配额'),
        ('if(j>0 && __any_sync', 'O correction 与 P stores 现在由同一个 owner 发布'),
        ('wait(base+FINAL,0)', '保护 causal 不同退出时刻的共享内存复用')])

lesson(13, 'mask fast path：完整 tile 不逐元素检查',
       '大部分非尾块和 causal 严格过去区域都无需 mask，旧代码仍对每个 score 做 bounds/select。',
       '先做 tile/当前行级判断：key tile 最后一个位置小于 seq，且 causal 时不超过 query，则跳过 128 次逐元素 mask。对 diagonal 和 tail 保留原慢路径。',
       '这是最终统一数据中较清楚的一步长序列收益。优先验证非 causal、causal、257/511 尾块，不能只测 4096 的整齐尺寸。',
       'fast path 必须同时满足 key bounds 和 causal 条件。TMA 越界填充值不等于 softmax 的 -infinity mask：零 score 仍会贡献概率。',
       'S=257、query=256 时，哪些 key tile 可跳过 mask，最后一个 tile 为什么不行？',
       'j=0 和 j=1 的全部 128 keys 有效；j=2 只有 key=256 有效，其余必须设为 -infinity。',
       [('if(j*128+127>=seq', '一次判断决定是否执行逐元素 mask')])

lesson(14, 'FMA 与数值稳定性：减少指令不保证更快',
       '每个 score 都计算 (score-reference)*scale，可尝试把 reference 缩放提到循环外。',
       '预计算 scaled_ref=-reference*scale，普通范围用 fmaf(score,scale,scaled_ref)。当 abs(reference)>65536 时保留先减后乘，使当前最大 score 的指数参数可靠地为零。',
       '最终统一复测中 v14 比 v13 慢，包括 S8192 non-causal；早期试跑曾见收益，不能用它覆盖最终结果。保留此课因为它是后续真实祖先，并展示指令数、编译器调度与数值误差的区别。',
       'scaled_ref 自身已舍入；FMA 不保证在 score==reference 时得到精确零。巨大的公共偏移会令中间 FP16 P 溢出，即使数学 softmax 不变。',
       '比较 fma(x,s,round(-x*s)) 与 (x-x)*s，为什么前者可能非零？',
       '第一种使用已舍入的公共项再融合乘加，不能消掉公共项的舍入误差；第二种先精确相减为零。检查中间 P，不能只分析最终归一化公式。',
       [('const float scaled_ref=', '把公共计算提出循环，同时引入公共舍入项'),
        ('if(fabsf(next)>65536.0f)', '数值稳定性分支与 FMA fast path 配套')])

lesson(15, 'max reduction：四条独立依赖链',
       '一条 max 链让下一次比较等待上一次结果；展开循环本身不自动消除这个依赖。',
       '把 128 个 scores 分配给 maxima[c%4]，最后两层合并四个结果。本课只改 max，rowsum 的求和顺序保持前版。',
       '与 v14 做配对比较。v10 的八链 max+sum 同时增加多个状态后回退，所以这里收缩为较小的单因素实验。ILP 收益必须结合寄存器和 spill 判断。',
       '所有初始化值必须为 -infinity；掩码 score 参与 reduction 时仍不能成为有效概率。',
       '估算一条 128 次串行 max 与四条各 32 次 max 的最长依赖深度。为什么不直接增加到 128 条链？',
       '四链大致是 32 次加两层合并；更多链会增加 live state 和合并成本，编译器/发射资源也有限。',
       [('float maxima[4]=', '四个 accumulator 分散依赖'),
        ('float mx=fmaxf(fmaxf', '最后使用短树合并')])

lesson(16, '寄存器预算：为 software owner 分配资源',
       'softmax 持有整行 scores、packed P 与 reduction 状态；metadata WG 的需求较小。静态统一配额与动态角色配额需共同考虑。',
       'entry __maxnreg__ 从 160 到 168，两个 softmax WGs 从 200 到 224；metadata 仍为 48。角色预算 256*224+128*48=63488，entry pool 为 384*168=64512。',
       '读 build.txt 的 spill 与 NCU local load/store，再看 latency。最终结果接近 v15，不把资源预算改善夸大为明显速度提升；后续 ILP 在这套预算上继续实验。',
       '总数检查只是必要的资源核算，还需满足 warpgroup、分配粒度和 subpartition 约束。反例 v34/v35 分别启动失败/停滞。',
       '为什么 NCU 的 launch registers=168 不代表 softmax 永远只能用 168 个寄存器？',
       '它反映入口编译配置；setmaxnreg 可在角色间重新分配。要结合指令、角色预算、SASS 和 spill 看实际情况。',
       [('__global__ __maxnreg__(168)', '入口池与软件角色的动态配额是两层概念'),
        ('setmaxnreg.inc.sync.aligned.u32 224;', '增加 softmax 配额前需由 metadata 角色释放预算')])

lesson(21, 'rowsum reduction：四条 pair-sum 链',
       'maximum 已有 ILP，psum 仍把 64 个 (a+b) 逐个串行累加。',
       '改为 sums[c%4]+=a+b，最后合并四项。maximum、exponent 和 buffer 调度保持 v16，独立测 rowsum。',
       '最终统一复测比 v16 的长序列更快。对照 v10 说明：同一个 ILP 想法放在不同资源预算和角色结构里，效果会不同。',
       '加法重新排序改变 FP32 rounding，重新检查独立参考误差；不要仅凭数学结合律认定浮点输出完全相同。',
       '为什么此处四条链只有四个标量 accumulator，而后面的 packed 版本需要八个？',
       '这里先标量 a+b 再累加；packed 版本分别累计 a 和 b 两个分量，一条 packed 链有两个 FP32 accumulator。',
       [('uint32_t packed[64];float sums[4]', '只改变 rowsum 的依赖结构'),
        ('const float psum=', '在更新 online rowsum 前合并四条链')])

lesson(23, 'causal 负载均衡：先安排较重的 query tiles',
       'causal 前部 query 只处理少量 keys，后部 query 处理更多。相同 tile 大小不代表相同 CTA 工作量；尾部剩下少数重 CTA 会拖长结束时间。',
       '从 v21 出发只加入 query-major reverse remap：把 linear CTA 坐标映射为逆序 query tile，batch/head 在 query 层内交错。non-causal 映射保持原状。',
       '独立 v22 实验中，NCU SM active cycles 的 max/mean 不均衡从约 +41% 降到约 +8%；v23 重新测组合，而非把两个历史 speedup 相乘。CUDA 不承诺严格按 blockIdx 顺序执行，这里是实测有效的映射。',
       'remap 只重新排列任务，必须双射覆盖每个 (batch,head,query tile)，不能交换输出内容或减少 causal 工作。',
       'gridDim.x=3、B*H=2 时，枚举 linear=0..5 对应的 query tile 和 bh。',
       '依次为 (2,0),(2,1),(1,0),(1,1),(0,0)。相同重 query 的两个 heads 提前进入映射。',
       [('const int linear=', '先线性化 grid，再映射 query-major 顺序'),
        ('const int qr=(CAUSAL?', 'causal 重任务优先；输出仍写真实 query 位置')])

lesson(24, '拆开 K/V 生命周期：尽早释放 K',
       'QK 只读 K，PV 才读 V；把它们绑在 KVFULL/KVEMPTY 上，意味着 QK 等不需要的 V，K 复用也等不需要的 PV。',
       '将 full/empty 分别拆成 KFULL、VFULL、KEMPTY、VEMPTY。最后一个 QK 后提交 KEMPTY，最后一个 PV 后提交 VEMPTY；loader 可先搬下一轮 K，再等 V slot。',
       'SMEM 实际仍是双 K slots、双 V slots。对照 v23 的 wait/commit 位置和配对 latency：收益来自解除多余依赖，不是增加容量。',
       'empty 是旧消费者完成，不是新 producer 已经开始。异步 MMA 发出后不能立即普通 arrive 宣称 K/V 可覆盖，要让 completion 释放。',
       '为什么 KEMPTY 不能放在第一个 Q-stage 的 QK 后？',
       '两个 Q stages 共享同一个 K tile；第二个 QK 还会读它，必须等最后一个实际消费者。',
       [('constexpr int QFULL=', '分别列出四种 K/V ownership barrier'),
        ('if(j>=2) wait(base+KEMPTY', 'loader 不再必须把 K 和 V 一起等待'),
        ('commit(base+KEMPTY+slot*8);', 'K 生命周期在两个 QK 使用结束后关闭')])

lesson(27, '交错 QK/PV：让另一个 stage 填补等待',
       '先发两个 QK、再集中等两个 softmax，仍可能留下 Tensor Core 空隙。双 Q-stage 只是提供了机会，还需选择合适 issue 顺序。',
       '改成 QK0(j) → PV1(j-1) → QK1(j) → PV0(j)。stage1 的 PV 延迟到下一轮；最后独立 drain 一次 PV1。只画稳态不够，还要画首轮和末轮。',
       'NCU TC active 从 v24 约 46.9% 到 v27 约 56.8%；第二轮同机长序列 latency 有明显下降。NCU 的 active 指标不是官方吞吐比例，最终排名用 CUDA Graph 数据。',
       'V(j) 必须保留到延后的 PV1(j) 完成。causal 两个 stage 的循环长度不同，stage0 的 if 不能误删 stage1 的工作；FINAL 在 drain 后。',
       '写出 j=0、j=1 和 loop 结束后的 issue 顺序，标出 V0 最早什么时候能复用。',
       'j=0 无 PV1(-1)；j=1 才执行 PV1(0)，其 completion 后 V0 可重用；最后追加 PV1(last)。',
       [('// Stagger stages:', '本课核心是 issue 顺序，不是更多 tile'),
        ('const int prev=j-1', '延后一轮的 PV1 读取前一 slot'),
        ('const int last=lengths[1]-1', '独立 drain 保护最后一个 stage1 输出')])

lesson(31, '消除多余等待：先证明 MMA 的读写顺序',
       'PV 把 P 作为 A 读取，下一 QK 把别名 S 作为 D 写入；旧实现用 PVDONE commit/wait 强制完成后再提交。',
       '同一 issuer warp、同一 cta_group 的这对 MMA 有 A-read→后续 D-write 顺序保证，所以去掉这条额外往返。SFULL 仍跟踪此前 MMA，使软件 correction 等到上一 PV 完成。',
       '本课结合 NVIDIA PTX pipelined tcgen05 规则与原 v31 的 96 项压力检查、六项 sanitizer 结果；参考链接见基础篇。sanitizer 是补充证据，不能替代对具体指令依赖的论证。',
       '只删除已证明冗余的依赖。PFULL、SFULL、K/V empty、FINAL、EPIDONE 和跨线程 fence 保留。不能推广为“同线程全部异步指令自动顺序完成”。',
       '为什么可以去掉 PVDONE，却不能去掉 softmax producer 的 PFULL？',
       '前者是同 warp MMA 的特定读写配对；后者跨 software owner 与 issuer，P 的 TMEM store 必须发布完成后才能被 PV 消费。',
       [('fence();qk(tm,base,kb);', '此处不再等待前次 PV 的额外 PVDONE'),
        ('wait(base+PFULL,j&1);fence();', '跨线程 P producer-consumer 交接仍然存在'),
        ('wait(base+SFULL+stage*8,j&1)', 'SFULL 也约束此前 O 的 PV 更新')])

lesson(42, 'packed FP32：同时减少指令与依赖',
       '标量 FMA/求和仍占 softmax 指令预算；直接打包但保留单一依赖链，又可能只是把瓶颈换个形式。',
       'argument2 用 f32x2 FMA，add_sum2 用 f32x2 add，分成四条 packed chains，也就是八个标量 accumulator；配合 max3 风格归约。max3 helper 源码是两条 max.f32，不应仅凭函数名认定为一条原生 PTX 指令。',
       '这是组合实验。v47 只改 max3 的消融接近持平，因此不能把 v42 的收益全部归给 max3。最终表有 case 差异；causal/non-causal 的取舍在下一课处理。',
       'f32x2 表示两个 FP32 运算，和将 P 转成 half2 是不同步骤。大 reference 稳定分支保留；psum 合并八项且每个分量只计一次。',
       '为什么把 64 次 packed add 全放进同一个 accumulator 可能仍然慢？',
       '每次 add 仍依赖前一次结果；四条链给调度器更多独立工作，同时需要更多寄存器，应结合 spill 看效果。',
       [('void argument2(', '两个 FP32 exponent 参数的 packed FMA'),
        ('void add_sum2(', '独立 packed sum chains 与减少指令同时考虑'),
        ('float max3(', '阅读实际 helper，不按名字推断机器指令数量')])

lesson(44, '按模式专门化：causal 与 non-causal 分开选择',
       '不同模式的循环长度、mask 和调度压力不同，一套 softmax 指令组织不一定兼顾所有 case。',
       'template<bool CAUSAL> 中用 if constexpr：non-causal 用 v42 packed 路径，causal 用 v31 标量 FMA 和四条 sum chains。选择在编译期完成，不增加每元素运行时模式判断。',
       '早期配对运行支持这个方向；最终统一表中 v42/v44 差异较小，不能写成每个 causal shape 都固定更快。v44 是 v51 大 grid 组件。',
       '分支改变计算组织，attention 定义、mask、scale 和稳定分支保持一致。最终结果重新验证，不能按组件历史数字推算组合结果。',
       'if constexpr 与读 tensor 内容后选路径有什么区别？',
       '前者按调用模式生成固定 specialization；本实现不读输入内容来选择算法，也不依赖是否处于 benchmark。',
       [('if constexpr(CAUSAL)', '同一个模板生成两种 softmax 指令组织'),
        ('const float psum=', 'causal 只合并四条有效标量链')])

lesson(39, '小任务支线：更高理论 occupancy 为什么仍慢',
       '回到 v31 分叉。双 Q-stage 使用较多 SMEM/TMEM，小 grid 的 CTA 数可能不足；尝试用更小 CTA 并行补偿。',
       '改为单 Q-stage、M128/N64、256 threads、256 TMEM columns，双 KV buffers。实际 N=64 由 qk descriptor、load_kv、循环步长和 tensor map 决定；原文件遗留 constexpr N=128 未驱动这些位置，不能仅看常量声明。',
       '理论资源允许两个 CTA，但 NCU achieved active warps 约 5.96、TC active 约 33.1%，长序列显著变慢。此课是诊断过渡，不是提速成功。下一课保留轻量结构、恢复 N128。',
       'N 变化会同时影响 score 数、P packing、PV reduction 长度、V descriptor stride、TMA box 和 causal/tail。不能只改一个 constexpr 就认为完成 tile 变换。',
       'N128→N64 时，访问相同 key 长度需要多少次循环？为什么资源更少却不一定更快？',
       '大致翻倍；更多同步、softmax 更新和 issue 固定开销，可能超过额外并发带来的收益。',
       [('void load_kv(', 'K/V 每次只搬 64 rows；和 Q 的 128 rows 区分'),
        ('((64/8)<<17)', '实际 QK N=64 编码在 descriptor'),
        ('fa4::encode(&ctx->k,k,batch,seq,heads,64)', 'host tensor map 也必须与 tile 一致')])

lesson(43, '小 grid 专用：恢复 N128，保留单 Q-stage',
       'N64 的循环与同步次数太多；可以恢复较大 key tile，同时用单 KV pair 控制容量。',
       '从 v39 恢复 N128，使用一个 Q tile、一个 K tile、一个 V tile；数据 SMEM 96 KiB 加 metadata，256 TMEM columns。softmax/metadata budget 调为 192/56。此课是 tile、buffer 与配额的联合设计，收益不能只归给 N。',
       '短序列明显快于 v31/v44，长序列 non-causal 却慢很多。30-shape sweep 表明，应按 CTA 数与设备规模选择，而非简单断言“S 越小用它越好”。',
       '单 buffer 也需分别保护 K 和 V 的生命周期；K 可在 QK 后先覆盖，V 必须等 PV 完成。只有一个 Q-stage，不要照搬双 stage 的下标含义。',
       'B=1,H=8 时，S512 和 S8192 的单 Q-stage CTA 数各是多少？',
       'ceil(S/128)*B*H，分别为 32 和 512。固定启动开销、填满 SM 的能力和多 wave 稳态成本的重要性不同。',
       [('constexpr int TILE_BYTES=', '容量来自一个 Q 和一个 K/V pair'),
        ('setmaxnreg.inc.sync.aligned.u32 192;', '单 Q-stage 的角色预算'),
        ('dim3 grid((c.seq+127)/128', '每 CTA 只负责 128 query rows')])

lesson(50, '修正 epilogue bank conflict：130 half → 129 half',
       'v43 输出 staging 是 FP16，读取 half2 时相邻 lanes 的 c 相差 2。pitch=130 half 使 bank 步长为 2，产生两路冲突。',
       '只把 staging 的写/读 pitch 全部由 130 改为 129。对固定 row，lane 每前进一次，地址跨 2*129 个 half，也就是 129 个 32-bit words，模 32 的步长为 1。',
       'NCU excessive shared wavefronts 从 v43 的 131072 降为 0；短序列 latency 只小幅下降，部分长序列还略慢。消除指标中的浪费不意味着端到端必有大提升。',
       '写入和读出必须使用一致 pitch，分配区也必须容纳最大下标。不要把第 3 课 FP32 staging 的 bank 推导直接套到 FP16。',
       '假设 pitch=128 half，同样 c 每次加 2，bank 步长是多少？',
       '128 个 32-bit words，模 32 为 0；这类访问会集中到同一个 bank，需要考虑实际指令和广播规则。',
       [('staging[(c*32+d)*129+local]', '写入 padding 与读取地址保持一致'),
        ('staging[c*129+r]', '按 half 的字节数推导 bank 映射')])

lesson(51, '汇合：按 CTA/SM 数选择完整 kernel',
       '大 grid 双 Q-stage 更擅长稳态吞吐，小 grid 单 Q-stage 更轻。单一版本无法在全部 shape 上取胜。',
       '完整组合 v44 与 v50。create 查询设备 SM 数；launch 计算 T=ceil(S/128)*B*H。T≤SM，或 causal 且 S≥2048 且 T≤2*SM 时用 small，其余用 large。每次仅 launch 一个完整 attention kernel。',
       '规则来自 30-shape sweep，v51 再经过统一 benchmark、120 项压力测试和两条路径共 12 项 sanitizer。推荐是覆盖范围内的折中，不是每个 case 的绝对最快版本；S8192 约达到官方吞吐 74%。',
       'tensor map 创建和设备查询不进入 kernel 计时。dispatch 只依赖 shape/mode/device；CUDA Graph 捕获选定路径后，shape 改变需要重新准备对应 context/graph。',
       '在 148 SM 的 B200 上，B1/H8/S4096 non-causal 与 causal 各走哪条路径？B1/H8/S8192 呢？',
       'S4096 的 T=256：non-causal 走 large，causal 满足 T≤296 走 small。S8192 的 T=512，两者都走 large。',
       [('namespace fa4_large', '大 grid 组件来自原 v44'),
        ('namespace fa4_small', '小 grid 组件来自原 v50'),
        ('struct DispatchContext', '分别构造 context，设备属性查询放在 launch 之外'),
        ('const int64_t small_tiles=', '本课真正新增的优化点：按工作量与设备规模选择')])
