"""Build annotated teaching copies and paired tables from existing audited runs.

Only tutorial/ is written. --check verifies provenance, generated files, and links.
No CUDA compilation, Modal calls, or new performance measurements are performed.
"""
import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path

from lessons import LESSONS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TAG = '// [教程] '


def digest(data):
    return hashlib.sha256(data).hexdigest()


def produce():
    originals = {x['version']: x for x in json.loads((ROOT / 'versions.json').read_text())}
    by_original = {x['original']: f'v{i:02}' for i, x in enumerate(LESSONS, 1)}
    runs = {name: json.loads((ROOT / 'runs' / name / 'results.json').read_text())
            for name in ['final_all_versions', 'r2_final_all']}
    files = {}
    manifest = []
    index = ['| 课号 | 原实验 | 本课主题 | 真实父版本 | 数据轮次 | S512 非 causal μs | S8192 非 causal μs | S8192 causal μs |',
             '| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
    for i, lesson in enumerate(LESSONS, 1):
        name = f'v{i:02}'
        original = lesson['original']
        origin = originals[original]
        parents = origin['parent'].split('+') if origin['parent'] else []
        raw = (ROOT / original / 'kernel.cu').read_bytes()
        assert digest(raw) == origin['source_sha256'], f'Changed audited source: {original}'
        code = raw.decode()
        assert TAG not in code
        lines = code.splitlines(keepends=True)
        inserts = {}
        for needle, explanation in lesson['anchors']:
            found = next((j for j, line in enumerate(lines) if needle in line), None)
            assert found is not None, (original, needle)
            inserts.setdefault(found, []).append(explanation)
        annotated = [f'{TAG}第 {i:02} 课：{lesson["title"]}；原实验 {original}。\n',
                     f'{TAG}真实父版本：{origin["parent"] or "无（基线）"}；完整说明见同目录 README.md。\n',
                     f'{TAG}仅增加中文导读，删除所有带此标记的注释行后，与已测原版逐字节一致。\n']
        locations = []
        for j, line in enumerate(lines):
            for explanation in inserts.get(j, []):
                annotated.append(f'{TAG}{explanation}\n')
                locations.append((len(annotated) + 1, explanation))
            annotated.append(line)
        annotated = ''.join(annotated)
        assert ''.join(line for line in annotated.splitlines(True) if not line.startswith(TAG)) == code
        files[f'{name}/kernel.cu'] = annotated
        patch_links = []
        for parent in parents:
            old = (ROOT / parent / 'kernel.cu').read_text()
            patch = ''.join(difflib.unified_diff(old.splitlines(True), code.splitlines(True),
                                                fromfile=f'{parent}/kernel.cu', tofile=f'{original}/kernel.cu'))
            files[f'{name}/from_{parent}.diff'] = patch
            patch_links.append(f'[{parent} → {original}](from_{parent}.diff)')
        run = 'final_all_versions' if int(original[1:]) <= 26 else 'r2_final_all'
        data = runs[run]
        chosen = [original, *parents]
        pairs = []
        for e in data['benchmarks']:
            for v in chosen:
                assert e['source_sha256'][v] == originals[v]['source_sha256']
                assert e['errors'][v]['passed']
                assert len(e['timing'][v]['samples_us']) == 9
            pairs.append(dict(shape=e['shape'], causal=e['causal'],
                              timing={v: e['timing'][v] for v in ['official_fa4', *chosen]},
                              errors={v: e['errors'][v] for v in chosen},
                              source_sha256={v: e['source_sha256'][v] for v in chosen},
                              clocks_before=e['clocks_before'], clocks_after=e['clocks_after']))
        evidence = dict(original=original, parents=parents, run=f'runs/{run}',
                        original_sha256=digest(raw), tutorial_sha256=digest(annotated.encode()),
                        source_result_sha256=digest((ROOT/'runs'/run/'results.json').read_bytes()),
                        timing=data['metadata']['timing'], benchmarks=pairs)
        files[f'{name}/evidence.json'] = json.dumps(evidence, ensure_ascii=False, indent=2) + '\n'
        parent_links = '、'.join(f'[{p}（第 {by_original[p][1:]} 课）](../{by_original[p]}/README.md)' for p in parents) or '无，串行基线'
        nav = '[路线总览](../README.md)'
        if i > 1:
            nav += f' · [上一课](../v{i-1:02}/README.md)'
        if i < len(LESSONS):
            nav += f' · [下一课](../v{i+1:02}/README.md)'
        md = [f'# 第 {i:02} 课：{lesson["title"]}', '', nav, '',
              f'对应原实验 **{original}**；真实父版本：{parent_links}。课号表示阅读顺序，父版本表示代码来源。', '',
              '## 本课要解决什么', '', lesson['problem'], '',
              '## 改法与原理', '', lesson['change'], '',
              '## 对照代码怎么读', '',
              '[完整中文导读源码](kernel.cu) · ' + (' · '.join(patch_links) if patch_links else '本课没有父版本 diff'), '',
              '先看以下标记，再按需展开 PTX 寄存器参数列表。diff 使用原实验源码比较，排除了新增教学注释。', '']
        md += [f'- [kernel.cu:{line}](kernel.cu#L{line})：{label}。' for line, label in locations]
        md += ['', '## 正确性与同步必须守住什么', '', lesson['invariant'], '',
               '## 如何读 profile 和性能结果', '', lesson['evidence'], '',
               f'下表均来自同一次 [{run}](../../runs/{run}/results.json)，B=1、H=8、D=128、FP16。',
               '父版/本版 >1 表示本版更快；<1 表示回退。不要跨运行直接相减。全部 8 个 shape、原始 9 轮样本、误差和源码哈希见 [evidence.json](evidence.json)。', '',
               '| S | causal | 本版 μs | 官方 μs | ' + ' | '.join(f'{p} μs / 父版÷本版' for p in parents) + (' |' if parents else ''),
               '| ---: | --- | ---: | ---: |' + ' ---: |' * len(parents)]
        for e in pairs:
            if e['shape'] not in [[1,512,8,128],[1,8192,8,128],[1,4096,8,128]]:
                continue
            if e['shape'][1] == 4096 and not e['causal']:
                continue
            t = e['timing']; own = t[original]['median_us']
            md.append(f'| {e["shape"][1]} | {e["causal"]} | {own:.3f} | {t["official_fa4"]["median_us"]:.3f} |' +
                      ''.join(f' {t[p]["median_us"]:.3f} / {t[p]["median_us"]/own:.3f}× |' for p in parents))
        md += ['', f'[原版实验说明](../../{original}/README.md) · [NCU 诊断](../../{original}/profile/ncu_details.txt) · '
               f'[NCU CSV](../../{original}/profile/ncu_raw.csv) · [编译与 spill](../../{original}/profile/build.txt)', '',
               '这些是原版源码的已保存测量。教学副本只新增注释，已校验正文一致；本次整理没有重新跑 GPU，也没有把教学副本声称为独立的新测量。', '',
               '## 动手检查', '', lesson['exercise'], '', '<details>', '<summary>参考思路</summary>', '', lesson['answer'], '', '</details>', '',
               '## 复现本课原实验', '', '仓库根目录，已配置 Modal 的环境中运行：', '', '```bash',
               f'.venv/bin/modal run kernel/fa4/run_modal.py --versions {",".join([*parents, original])} --mode benchmark',
               f'.venv/bin/modal run kernel/fa4/run_modal.py --versions {original} --mode profile', '```', '',
               '修改教学副本后如何测试，见 [实验操作指南](../LAB.md)。上述命令测的是原实验目录，不会自动测你的教学副本修改。', '']
        files[f'{name}/README.md'] = '\n'.join(md)
        values = [next(e['timing'][original]['median_us'] for e in pairs
                       if e['shape']==[1,s,8,128] and e['causal']==causal)
                  for s, causal in [(512,False),(8192,False),(8192,True)]]
        index.append(f'| [{name}]({name}/README.md) | [{original}](../{original}/README.md) | {lesson["title"]} | '
                     f'{origin["parent"] or "—"} | {"R1" if run=="final_all_versions" else "R2"} | ' +
                     ' | '.join(f'{x:.2f}' for x in values) + ' |')
        manifest.append(dict(lesson=name, original=original, parents=parents, title=lesson['title'],
                             original_sha256=digest(raw), tutorial_sha256=digest(annotated.encode()),
                             evidence=f'{name}/evidence.json', anchors=locations))
    files['manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    readme = (HERE/'README.md').read_text()
    start, end = '<!-- ROUTE_TABLE_START -->', '<!-- ROUTE_TABLE_END -->'
    assert readme.count(start)==readme.count(end)==1
    files['README.md'] = readme.split(start)[0]+start+'\n\n'+'\n'.join(index)+'\n\n'+end+readme.split(end)[1]
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    files = produce()
    for relative, content in files.items():
        path = HERE/relative
        if args.check:
            assert path.read_text() == content, f'Generated file changed: {path}'
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    for md in HERE.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', md.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            path, _, fragment = target.partition('#')
            linked = md.parent/path
            assert linked.exists(), (md, target)
            if fragment.startswith('L') and fragment[1:].isdigit():
                assert 1 <= int(fragment[1:]) <= len(linked.read_text().splitlines())
    print(f'{"Verified" if args.check else "Built"} {len(LESSONS)} lessons; '
          f'{len(files)} generated files; code bodies, paired samples, hashes and local links verified.')


if __name__ == '__main__':
    main()
