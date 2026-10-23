"""Audit immutable final runs and publish per-version artifacts + a Chinese report."""
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BENCH=ROOT/'runs/final_all_versions'
PROFILE_BATCHES=['01_05','06_10','11_15','16_20','21_25','26']
META=[
(None,'串行 TMA/TC/TMEM 基线','基线'),
('v1','直接 PTX exp2','保留'),('v2','合并 output stores','保留'),
('v3','conditional rescaling','小幅收益'),('v4','每行一次 reciprocal','保留'),
('v5','双 K/V buffer 预取','短序列候选'),('v5','双 Q / 四 WG roles','教程结构对照'),
('v7','50% polynomial exponent','回退，保留实验'),('v7','warp arrivals','无稳定收益'),
('v7','8 条 max/sum chains','回退'),('v7','P split completion','无稳定收益'),
('v7','correction / epilogue 合并','保留'),('v12','valid-tile mask fast path','保留'),
('v13','FMA argument + 极大 reference 稳定分支','保留'),('v14','4 条 max chains','保留'),
('v15','168/224 register budget','保留'),('v16','单条 x128 score load','回退'),
('v16','单 KV slot','回退'),('v16','packed FMA/add','non-causal 回退'),
('v16','pair-first max3','无稳定收益'),('v16','4 条 rowsum chains','保留'),
('v16','causal query-major reverse','保留'),('v21','加入 causal reverse','保留'),
('v23','K/V readiness / lifetime 分离','主要推荐'),('v24','5-slot K/V ring','无额外收益'),
('v11','四 WG / P split / mixed exp / TMA output 整合','教程 non-causal 路径对照'),
]

def hashfile(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run_url(p):
    if not p.exists():return ''
    m=re.search(r'https://modal.com/apps/\S+',p.read_text())
    return m.group(0) if m else ''

assert (BENCH/'exit_code.txt').read_text().strip()=='0'
data=json.loads((BENCH/'results.json').read_text())
assert len(data['correctness'])==1040 and all(e['error']['passed'] for e in data['correctness'])
assert len(data['benchmarks'])==8
versions=[f'v{i}' for i in range(1,27)]
for e in data['benchmarks']:
    assert set(e['source_sha256'])==set(versions)
    for v in versions:assert e['source_sha256'][v]==hashfile(ROOT/v/'kernel.cu')
profiles={}
for label in PROFILE_BATCHES:
    run=ROOT/'runs'/f'final_profile_{label}'
    if not run.exists():continue
    for v in versions:
        p=run/v
        if (p/'profile.ncu-rep').exists():profiles[v]=p
assert set(profiles)==set(versions),f'Missing profiles: {set(versions)-set(profiles)}'
for v,p in profiles.items():
    assert hashfile(p/'kernel.cu')==hashfile(ROOT/v/'kernel.cu'),f'{v} profile source mismatch'
    assert (p/'ncu.exit_code').read_text().strip()=='0',f'{v} NCU failed'

san=ROOT/'runs/final_sanitizers'
assert len(list(san.rglob('*.exit_code')))==18
assert all(p.read_text().strip()=='0' for p in san.rglob('*.exit_code'))
manifest=[]
for i,(parent,change,decision) in enumerate(META,1):
    v=f'v{i}';dest=ROOT/v
    own=dict(metadata=data['metadata'],source_sha256=hashfile(dest/'kernel.cu'),
             benchmark_run=str(BENCH.relative_to(ROOT)),profile_run=str(profiles[v].parent.relative_to(ROOT)),
             correctness=[e for e in data['correctness'] if e['version']==v],benchmarks=[])
    for e in data['benchmarks']:
        own['benchmarks'].append(dict(shape=e['shape'],causal=e['causal'],timing={k:e['timing'][k] for k in ['official_fa4',v]},
                                      error=e['errors'][v],clocks_before=e.get('clocks_before'),clocks_after=e.get('clocks_after')))
    (dest/'results.json').write_text(json.dumps(own,indent=2)+'\n')
    pd=dest/'profile';pd.mkdir(exist_ok=True)
    for p in profiles[v].iterdir():
        if p.name in ['build.txt','sass.txt','ncu.log','ncu.exit_code','ncu_details.txt','ncu_raw.csv','ncu_source.txt','profile.ncu-rep','kernel.cu'] or p.suffix in ['.ptx','.cubin']:
            shutil.copy2(p,pd/p.name)
    extra=['','## 最终统一复测','',f'父版本：{parent or "无"}；决定：{decision}。',
           '当前源码、profile 源码与统一 benchmark 的 SHA256 已逐项核对。40 项 FP32 正确性检查全部通过。',
           '完整原始数据：[results.json](results.json)，[NCU report](profile/profile.ncu-rep)，[NCU details](profile/ncu_details.txt)，[SASS](profile/sass.txt)。',
           '', '| B,S,H,D | causal | CUDA us | 官方 us | 官方吞吐 % |','| --- | --- | ---: | ---: | ---: |']
    for e in own['benchmarks']:
        t=e['timing'][v];o=e['timing']['official_fa4']
        extra.append(f"| {','.join(map(str,e['shape']))} | {e['causal']} | {t['median_us']:.3f} | {o['median_us']:.3f} | {t['official_throughput_percent']:.2f} |")
    p=dest/'README.md';text=p.read_text().split('\n## 最终统一复测')[0];p.write_text(text+'\n'+'\n'.join(extra)+'\n')
    manifest.append(dict(version=v,parent=parent,change=change,decision=decision,source_sha256=own['source_sha256'],profile_source_sha256=hashfile(profiles[v]/'kernel.cu'),correctness_checks=40,profile_run=own['profile_run']))
(ROOT/'versions.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')

# Keep paired data and clearly distinguish standard input comparisons from stress tests.
normal=[e for e in data['correctness'] if e['distribution']!='constant_large_logits']
extreme=[e for e in data['correctness'] if e['distribution']=='constant_large_logits']
assert all(e['official_error']['passed'] for e in normal)
official_extreme_fail=sum(not e['official_error']['passed'] for e in extreme)
lines=['# Profiling 驱动的优化记录','',
       '所有最终版本在同一 Modal B200 上统一复测，FP16 输入/输出、FP32 accum、D=128。当前源码全部与运行快照 SHA256 匹配。',
       f'CUDA Graph：20 nodes / replay，10 轮预热，9 轮交替顺序，CUDA events 中位数；共 1040 项 FP32 正确性检查、208 个 version/shape 性能结果。',
       '', '## 最终主结果','', '| Shape B,S,H,D | causal | v1 us | v24 us | 官方 FA4 us | v1→v24 | v24 官方吞吐 % |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for e in data['benchmarks']:
    t=e['timing'];a=t['v1']['median_us'];b=t['v24']['median_us'];o=t['official_fa4']['median_us']
    lines.append(f"| {','.join(map(str,e['shape']))} | {e['causal']} | {a:.3f} | {b:.3f} | {o:.3f} | {a/b:.3f}× | {100*o/b:.2f} |")
lines+=['','长序列主要推荐 v24；512/1024 短序列 v6 更快。需要阅读四 WG / split P / mixed exponent / TMA store 的教程路径，查看 v26，不能把版本号当作性能排名。',
        '', '## 每轮实验与取舍','', '| Version | Parent | 修改 | 决定 |','| --- | --- | --- | --- |']
for m in manifest:lines.append(f"| [{m['version']}]({m['version']}/README.md) | {m['parent'] or '—'} | {m['change']} | {m['decision']} |")
lines+=['','## 关键 profile 证据','',
        'v1 的 output stores 有 3,932,160 excessive sectors，占 4,194,304 sectors 的约 94%；合并输出后降为 0。',
        'PTX/SASS 显示每行 128 次 div.rn，以及通用 exp2f 的范围处理；改为直接 exp2 与一次 reciprocal 后去掉不必要指令。',
        '完整 tile 的 mask fast path 去掉重复 bounds/select；4-chain maximum 与 rowsum 的较小 ILP 在新 register budget 下有效，而旧 8-chain 同时修改两种 reduction 没有收益。',
        'v16 causal S=8192 的 SM active cycles 最大高于均值 41.38%、最小低 36.51%；v22 query-major reverse 后降为 +8.35% / −4.36%。该改动的同机 causal latency 从约 309 降到 200 us。',
        'K/V 生命周期分离能小幅改善 overlap；再增加到 5-slot ring 没有额外稳定收益。Mixed polynomial、P split、warp arrivals、x128 load 均保留，但不能因技术来自论文就宣称在本实现里加速。',
        '', '## 正确性与同步','',
        '40-case / version：B=1 的 S=128/256/257/511/H=2，以及 B=2/S=257/H=3；causal/non-causal；normal、zero、large logits、constant large logits。独立 FP32 matmul/softmax 参考关闭 TF32；保存 max_abs/RMSE/relative L2。性能形状另对照官方输出。',
        'v24/v25/v26 在 S=511、causal/non-causal 上完成 memcheck/synccheck/racecheck：18 项检查均 exit 0，零错误和零警告，覆盖 KV ring reuse 与 TMA output 的边界。v6/v7/v12/v13 的早期审计也保留。',
        'v12 首轮 causal 的 future-parity 等待失败已保留，修正为独立 FINAL completion。v1-v8 补齐 TC before-thread-sync fence。',
        f'恒定超大 Q/K=30000 暴露了 FMA 公共 reference rounding shift 导致 FP16 P 溢出。FMA 版本现有 abs(reference)>65536 的先减再乘稳定分支。本次官方 b35 在该压力案例的记录中出现非有限输出（{official_extreme_fail} 条重复 version 对照记录）；它们不用于性能排名。官方在普通、零、large_logits=6 的对照全部通过。',
        '', '## 本轮停止点与限制','',
        '最终 v24 相比更深 ring v25 和整合版 v26 更快。最后对 polynomial、packed arithmetic、max3、额外 buffer 与 TMA load 形状的尝试没有找到全面、可复现的下一项加速；本轮停止在这个经过验证的实现。没有证明全局最优，也没有达到官方 FA4 的速度。',
        '范围仅 dense self-attention FP16 forward、head_dim=128、contiguous BSHD；没有 backward/dropout/GQA/varlen/FP8。v26 对照教程重点讲解的 non-causal 四角色数据路径；其 causal epilogue 仍放在 WG2，与文章 specialization 不同。',
        '剩余主要差距包括 compiler register/spill、softmax/TMEM 调度和数据搬运布局。更大 MMA tile、多 CTA 或重新组织输入 TMA 布局都需要新的布局与同步设计，不能把未测的推断写成已完成的优化。',
        '', '## 原始运行与重现','',
        f"[同机总复测]({run_url(ROOT/'final_benchmark.log')})；[最终 sanitizer]({run_url(ROOT/'final_sanitizers.log')})。",
        '第一次全量 final profile 回传因网络 Connection lost 中断，失败日志保留；最终报告来自小批次重新运行，逐版本复制到 profile/，完整批次位于 runs/。',
        '', '```bash',
        '.venv/bin/modal run kernel/fa4/run_modal.py --versions v24,v26 --mode benchmark',
        '.venv/bin/modal run kernel/fa4/run_modal.py --versions v24 --mode profile',
        '.venv/bin/modal run kernel/fa4/run_modal.py --versions v24 --mode profile --seq 8192 --causal',
        '.venv/bin/modal run kernel/fa4/run_modal.py --versions v24,v26 --mode sanitize',
        '.venv/bin/python kernel/fa4/summarize.py','```']
(ROOT/'OPTIMIZATION_LOG.md').write_text('\n'.join(lines)+'\n')
print('Audited and packaged 26 versions, 1040 correctness checks, 208 benchmark rows, 26 NCU reports, 18 final sanitizer checks.')
