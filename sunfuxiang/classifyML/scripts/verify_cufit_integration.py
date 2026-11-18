from pathlib import Path
import json,csv,hashlib,re

repo=Path('/home/login/data/cle/sunfuxiang/classifyML')
dest=repo/'cufit_comparison_results'
audit=json.loads((dest/'INTEGRATION_AUDIT.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for change in audit['changes']:
    assert sha(change['path'])==change['after_sha256']
    assert sha(change['backup'])==change['before_sha256']
for source in audit['prediction_archives']:
    assert sha(dest/source['archive'])==source['sha256']
text=(repo/'analysis_3seed_summary.md').read_text()
tables=[];rows=[]
for line in text.splitlines()+['']:
    if line.startswith('|'):
        rows.append(line)
    elif rows:
        widths=[len(row.split('|'))-2 for row in rows]
        assert len(set(widths))==1,(widths,rows[-1])
        tables.append(dict(columns=widths[0],rows=len(rows)-2));rows=[]
assert len(tables)==10,tables
overall=list(csv.DictReader((repo/'analysis_3seed_overall.csv').open()))
perclass=list(csv.DictReader((repo/'analysis_3seed_perclass.csv').open()))
assert len(overall)==14 and len(perclass)==42
assert len([r for r in overall if r['model']=='CUFIT' and r['dataset']=='full'])==1
assert len([r for r in perclass if r['model']=='CUFIT' and r['dataset']=='full'])==3
assert not any(r['model']=='CUFIT' and r['dataset']=='hq' for r in overall)
report=dest/'COMPARISON_REPORT.md'
content=report.read_text(encoding='utf-8')
content=content.replace('../EXPERIMENT_PLAN.md','EXPERIMENT_PLAN.md').replace('../audit.json','audit.json').replace('../references.json','references.json')
report.write_text(content,encoding='utf-8')
(dest/'README.md').write_text('''# CUFIT Full 对比归档

主汇总：`../analysis_3seed_summary.md`（含CUFIT的Full主指标、补充总体指标、分类别指标、三个种子明细及表9完整方案对比）。

详细报告： [COMPARISON_REPORT.md](COMPARISON_REPORT.md)。

- SUMMARY.csv / PER_SEED.csv / PER_CLASS.csv / RESULTS.json：CUFIT、Ours与RVC/RC参考组的统计，数值为0–1尺度。
- CUFIT_FULL_SEED_METRICS.csv：额外补齐MacroSpec、AUC-PR、BalancedAcc等指标。
- predictions/：本次汇总用到的12份逐样本预测副本。
- PREDICTION_SOURCES.json：预测原路径、副本及SHA256。
- audit.json / execution_files_sha256.json / references.json / EXPERIMENT_PLAN.md：数据、代码、权重与方案来源。
- INTEGRATION_AUDIT.json：主总结更新记录，以及更新前的精确备份路径和校验值。
- integration_backups/：修改前4份总结文件；仅用于回溯，不代表当前结果。

训练权重与完整日志保存在 `/home/login/data/cle/sunfuxiang/CUFIT0613/experiment_full_vs_ours_20261022/`，没有重复复制大权重。

主结论：CUFIT Full Macro-F1 90.43±0.38%，HQ + Ours 91.05±1.00%，后者描述性均值高0.619个百分点；未做显著性检验。
''',encoding='utf-8')
validation=dict(passed=True,markdown_tables=tables,overall_rows=len(overall),perclass_rows=len(perclass),
                original_backups_verified=True,legacy_csv_rows_preserved=True,predictions_verified=12,
                integrated_files_sha256={c['path']:sha(c['path']) for c in audit['changes']},
                archive_report_sha256=sha(report))
(dest/'INTEGRATION_VERIFICATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(validation,ensure_ascii=False,indent=2))
