"""Build auditable markdown tables from saved Modal JSON/NCU artifacts."""
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=[]
for run in sorted((ROOT/'runs').iterdir()):
    p=run/'results.json'
    if not p.exists():continue
    data=json.loads(p.read_text())
    for e in data.get('benchmarks',[]):
        official=e['timing']['official_fa4']['median_us']
        for v,t in e['timing'].items():
            if v=='official_fa4':continue
            source=ROOT/v/'kernel.cu'
            matches=hashlib.sha256(source.read_bytes()).hexdigest()==e['source_sha256'][v]
            rows.append(dict(run=run.name,version=v,shape=e['shape'],causal=e['causal'],us=t['median_us'],official_us=official,percent=100*official/t['median_us'],tflops=t['tflops'],sha_matches=matches))
(ROOT/'measurements.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# 保存的性能记录','', '每行来自一次同机 paired run；不同运行的 GPU clocks 可能不同。官方百分比 = official_us/custom_us × 100。', '', '| Run | Version | B,S,H,D | causal | CUDA us | official us | 官方吞吐 % | TFLOPS | 当前源码 SHA 匹配 |','| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
for e in rows:
    lines.append(f"| [{e['run']}](runs/{e['run']}/results.json) | {e['version']} | {','.join(map(str,e['shape']))} | {e['causal']} | {e['us']:.3f} | {e['official_us']:.3f} | {e['percent']:.2f} | {e['tflops']:.1f} | {e['sha_matches']} |")
lines+=['','## NCU 对照','', 'NCU 开启 replay，会改变 cache 状态；以下时间仅用于诊断，不用于排名。','', '| Version | Run | NCU us | registers | Tensor pipe elapsed % | excess store sectors |','| --- | --- | ---: | ---: | ---: | ---: |']
for run in sorted((ROOT/'runs').iterdir()):
    for p in sorted(run.glob('*/ncu_raw.csv')):
        r=list(csv.reader(p.open()))
        if len(r)<3:continue
        d=dict(zip(r[0],r[2]));v=p.parent.name
        lines.append(f"| {v} | [{run.name}]({p.relative_to(ROOT)}) | {d.get('gpu__time_duration.sum','?')} | {d.get('launch__registers_per_thread','?')} | {d.get('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','?')} | {d.get('derived__memory_l2_theoretical_sectors_global_excessive','?')} |")
lines+=['','## 失败与未完成运行','',
        '保留失败尝试，不纳入有效版本排名。部分运行在产出部分数据后失败，详见 execution.log 和子目录退出码。','',
        '| Run | 失败证据 |','| --- | --- |']
for run in sorted((ROOT/'runs').iterdir()):
    failures=[]
    for p in [run/'exit_code.txt', *sorted(run.rglob('*.exit_code'))]:
        if p.exists() and p.read_text().strip()!='0':
            failures.append(f'[{p.relative_to(run)}]({p.relative_to(ROOT)}) = {p.read_text().strip()}')
    if (run/'FAILED.txt').exists():failures.append(f'[FAILED.txt]({(run/"FAILED.txt").relative_to(ROOT)})')
    if not (run/'results.json').exists() and not (run/'exit_code.txt').exists() and not failures:
        failures.append('未保存完整运行结果，不能视为通过')
    if failures:lines.append(f'| [{run.name}]({run.relative_to(ROOT)}/) | '+ '; '.join(failures)+' |')
(ROOT/'MEASUREMENTS.md').write_text('\n'.join(lines)+'\n')
print(f'{len(rows)} benchmark rows saved')
