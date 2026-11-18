"""Integrate audited CUFIT Full seeds 42/43/44 into classifyML's existing summaries.

Run with the sunfuxiang Python environment. No training; refuses duplicate rows.
"""
from pathlib import Path
from datetime import datetime
import csv, io, json, hashlib, shutil, os, sys, re
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

ROOT=Path('/home/login/data/cle/sunfuxiang')
REPO=ROOT/'classifyML'
EXP=ROOT/'CUFIT0613/experiment_full_vs_ours_20261022'
DEST=REPO/'cufit_comparison_results'
CLASSES=['Benign_lesions','Malignant_lesions','body_of_stomach_normal']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o): Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def fmt(values):
    values=np.asarray(values,dtype=float)*100
    assert values.shape==(3,) and np.isfinite(values).all()
    return f'{values.mean():.2f}±{values.std(ddof=1):.2f}'
def append_csv(original,newrows):
    fieldnames=next(csv.reader(io.StringIO(original)))
    s=io.StringIO(newline=''); w=csv.DictWriter(s,fieldnames=fieldnames,lineterminator='\r\n')
    w.writerows(newrows)
    return original+('' if original.endswith('\n') else '\r\n')+s.getvalue()
def add_table_rows(text,heading,rows):
    start=text.index(heading); stop=text.find('\n## ',start+1)
    if stop<0: stop=len(text)
    section=text[start:stop]
    pipes=list(re.finditer(r'^\|.*\|$',section,re.M))
    assert pipes,heading
    pos=start+pipes[-1].end()
    return text[:pos]+'\n'+'\n'.join(rows)+text[pos:]

def main():
    assert json.loads((EXP/'status.json').read_text())['state']=='complete'
    sys.path.insert(0,str(EXP))
    from report import metrics,build
    assert build(), 'Incomplete experiment'
    result=json.loads((EXP/'results/RESULTS.json').read_text())
    assert result['complete']
    per_seed=[]; per_class=[]
    for seed in (42,43,44):
        base=EXP/f'checkpoints/full_seed{seed}'
        marker=json.loads((base/'evaluation_validated.json').read_text())
        assert marker['predictions_sha256']==sha(base/'test_predictions.csv')
        assert marker['checkpoint_sha256']==sha(base/'model_best.pth.tar')
        m,classes=metrics(base/'test_predictions.csv')
        raw=json.loads((base/'test_results.json').read_text())
        for a,b in [('accuracy','accuracy'),('macro_f1','f1_macro'),('macro_auroc_ovr','auroc_ovr_macro')]:
            assert abs(m[a]-raw[b])<1e-6
        pred=list(csv.DictReader((base/'test_predictions.csv').open()))
        y=np.array([CLASSES.index(r['true_label']) for r in pred])
        yp=np.array([CLASSES.index(r['pred_label']) for r in pred])
        probs=np.array([[float(r['prob_'+c]) for c in CLASSES] for r in pred])
        cm=confusion_matrix(y,yp,labels=[0,1,2])
        specs=[]
        for i,pc in enumerate(classes):
            tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp; tn=cm.sum()-tp-fp-fn
            specs.append(float(tn/(tn+fp)))
            per_class.append(dict(seed=seed,**pc))
        ap=float(np.mean([average_precision_score(y==i,probs[:,i]) for i in range(3)]))
        per_seed.append(dict(seed=seed,**m,balanced_acc=m['macro_recall'],macro_spec=float(np.mean(specs)),auc_pr=ap))
    mapping={'Macro-F1':'macro_f1','Accuracy':'accuracy','AUROC':'macro_auroc_ovr','Weighted-F1':'weighted_f1',
             'BalancedAcc':'balanced_acc','MacroPrec':'macro_precision','MacroSens':'macro_recall',
             'MacroSpec':'macro_spec','AUC-PR':'auc_pr','Kappa':'kappa','MCC':'mcc'}
    overall=dict(model='CUFIT',dataset='full',**{label:fmt([r[key] for r in per_seed]) for label,key in mapping.items()})
    class_rows=[]
    for name in CLASSES:
        rr=[r for r in per_class if r['class_name']==name]
        class_rows.append(dict(model='CUFIT',dataset='full',**{'class':name.replace('_',' ')},
              **{label:fmt([r[key] for r in rr]) for label,key in [('Sens','recall'),('Spec','specificity'),('Prec','precision'),('F1','f1'),('AUROC','auroc')]}))
    targets=['analysis_3seed_summary.md','analysis_3seed_overall.csv','analysis_3seed_perclass.csv','PROJECT_DOCS.md']
    originals={name:(REPO/name).read_bytes() for name in targets}
    oldtexts={name:data.decode('utf-8') for name,data in originals.items()}
    legacy_overall=list(csv.DictReader(io.StringIO(oldtexts['analysis_3seed_overall.csv'])))
    legacy_classes=list(csv.DictReader(io.StringIO(oldtexts['analysis_3seed_perclass.csv'])))
    assert not any(r['model']=='CUFIT' for r in legacy_overall), 'CUFIT already integrated; review instead of duplicating'
    text=oldtexts['analysis_3seed_summary.md']
    assert 'CUFIT' not in text,'Unexpected prior CUFIT text'
    text=text.replace('（均值±标准差，含 Ours）','（均值±标准差，含 Ours 与 CUFIT）',1)
    text=text.replace('- 基线模型 6 个 + **Ours（ResNet34+DegConv+CPCA，本工作）**',
        '- 原基线模型 6 个 + **Ours（ResNet34+DegConv+CPCA，本工作）** + **CUFIT（DINOv2 ViT-S/14 + 双 Rein）**',1)
    text=text.replace('**Ours 仅在 HQ 上训练/评估**（无 Full 结果处以 “—” 表示）',
        '**Ours 仅报告 HQ 训练结果，CUFIT 仅报告 Full 训练的 3-seed 结果**；各组在同一独立 test 集评估，未报告项以 “—” 表示',1)
    text=text.replace('随机种子 42/43/44，每模型每数据集各 3 次；',
        '随机种子 42/43/44，每个已报告的模型/训练数据组合各 3 次；',1)
    anchor='- <span style="color:red">**红色数字 = HQ 相对 Full 的提升（单位：百分点）**</span>'
    text=text.replace(anchor,anchor+'\n- CUFIT：Full 训练集 4977 张；三方共识 HQ 训练集 2880 张；共享 val=538、test=671。seed 42 复测后复用，43/44 于 2026-10-22 补跑。详见[完整对比报告](cufit_comparison_results/COMPARISON_REPORT.md)。',1)
    text=add_table_rows(text,'## 表1',[
        f'| **CUFIT** | {overall["Macro-F1"]} | {overall["Accuracy"]} | {overall["AUROC"]} | — | — | — | — |'])
    text=text.replace('Δ = HQ均值 − Full均值。Ours 无 Full 版本故为 “—”。',
        'Δ = HQ均值 − Full均值。Ours 无 Full 版本、CUFIT 无 HQ 三种子版本，故为 “—”。基线平均仅统计原有 6 个具备配对结果的模型。',1)
    text=add_table_rows(text,'## 表2',['| **CUFIT** | '+' | '.join(['—']*11)+' |'])
    text=add_table_rows(text,'## 表3',[
        '| **CUFIT** | '+' | '.join(overall[k] for k in ['Weighted-F1','BalancedAcc','MacroPrec','MacroSens','MacroSpec','AUC-PR','Kappa','MCC'])+' |'])
    text=add_table_rows(text,'## 表5',[
        '| **CUFIT** | '+r['class']+' | '+' | '.join(r[k] for k in ['Sens','Spec','Prec','F1','AUROC'])+' |' for r in class_rows])
    text=add_table_rows(text,'## 表8',[
        '| **CUFIT** | 未测 | 未测 | 未测 | 未测 / — | 未测 / — | 未测 / — |'])
    text=text.replace('FPS 为 3-seed 均值（RTX 4090）。Ours 无 Full → Full 列 “—”',
        'FPS 为原有 3-seed 均值（RTX 4090）。Ours 无 Full → Full 列 “—”；CUFIT 本次未测复杂度/推理效率，不从训练耗时推算 FPS',1)
    detail=[]
    for i,(label,key) in enumerate([('Macro-F1','macro_f1'),('Accuracy','accuracy'),('AUROC','macro_auroc_ovr')]):
        detail.append('| '+('**CUFIT**' if i==0 else ' ')+' | '+label+' | '+
                      ' | '.join(f'{r[key]*100:.2f}' for r in per_seed)+' | — | — | — |')
    text=add_table_rows(text,'## 附录',detail)
    summaries={r['method']:r for r in result['summary']}
    a=summaries['CUFIT + Full']; b=summaries['HQ + Ours']
    df=(b['macro_f1_mean']-a['macro_f1_mean'])*100
    da=(b['accuracy_mean']-a['accuracy_mean'])*100
    du=(b['macro_auroc_ovr_mean']-a['macro_auroc_ovr_mean'])*100
    comparison=['## 表9　CUFIT Full 与三方共识完整方案对比（2026-10-22）','',
        '两条方案均从同一份 Full 原始训练集出发：CUFIT 直接训练 Full；本方法先用三方 OOF 共识筛出 HQ，再训练 Ours。seed 均为 42/43/44，共享相同 val/test，统计口径为均值±样本标准差。','',
        '| 方案 | 实际训练图像 | Macro-F1 | Accuracy | AUROC |',
        '|:--|--:|--:|--:|--:|']
    for name in ['CUFIT + Full','HQ + Ours','RVC + ResNet34 (reference)','RC + ResNet34 (reference)']:
        r=summaries[name]
        label=name.replace(' (reference)','（消融参考）')
        v=[f'{r[k+"_mean"]*100:.2f}±{r[k+"_std"]*100:.2f}' for k in ['macro_f1','accuracy','macro_auroc_ovr']]
        comparison.append('| '+label+f' | {r["train_size"]} | '+' | '.join(v)+' |')
    comparison+=['',f'**主比较：HQ + Ours 相对 CUFIT Full 的 Macro-F1、Accuracy、AUROC 分别高 {df:.3f}、{da:.3f}、{du:.3f} 个百分点。**'
         '这是三种子的描述性均值差异，未作显著性检验；不从单个最高种子得出方法优劣结论。', '',
         '- RVC 为三方共识，RC 为双方共识，参考组不替代主比较。',
         '- CUFIT 未补跑 HQ seed 43/44；历史 HQ seed 42 单次记录不混入三种子汇总，也不计算 CUFIT 的 HQ−Full 提升。',
         '- 比较差异包含筛选、模型结构、预训练和训练/预处理策略，不能全部归因于三方共识；患者级划分本次未独立复核。',
         '- CUFIT 保留 Resize(256×256)+CenterCrop(224)，Ours 保留短边 Resize(256)+CenterCrop(224)。',
         '- seed 42 CUFIT 与三个 Ours 权重复测主要指标均与历史记录一致；旧 seed 42 的完整训练环境无法追溯。',
         '- CUFIT/Ours 主比较使用本次高精度预测；RVC/RC 参考沿用历史 6 位小数预测，AUROC 可能有舍入差异。',
         '- [完整报告](cufit_comparison_results/COMPARISON_REPORT.md) · [逐种子](cufit_comparison_results/PER_SEED.csv) · [总体](cufit_comparison_results/SUMMARY.csv) · [分类别](cufit_comparison_results/PER_CLASS.csv) · [CUFIT 补充逐种子指标](cufit_comparison_results/CUFIT_FULL_SEED_METRICS.csv)。','']
    text=text.replace('## 附录', '\n'.join(comparison)+'\n## 附录',1)
    docs=oldtexts['PROJECT_DOCS.md']
    docs=docs.replace('（8 张表）：6 基线', '（新增 CUFIT Full 与表9，共 9 张表及种子附录）：6 基线',1)
    old='| `analysis_3seed_overall.csv` / `analysis_3seed_perclass.csv` | 上文的机器可读版（总体 / 每类指标，mean±std） |'
    assert old in docs
    docs=docs.replace(old,old+'\n| `cufit_comparison_results/` | **CUFIT Full 补齐 seed 43/44，seed 42 复测复用**；3-seed Macro-F1 90.43±0.38 / Acc 90.31±0.39 / AUROC 97.97±0.03；与 HQ+Ours 主比较及 RVC/RC 参考，含详细报告、CSV、审计和原始预测归档 |',1)
    docs+='\n\n## 17. CUFIT Full 对比补充（2026-10-22）\n\n'+\
         f'CUFIT Full：Macro-F1 {overall["Macro-F1"]}%，Accuracy {overall["Accuracy"]}%，AUROC {overall["AUROC"]}%；'
    docs+=f'HQ + Ours 的三个种子平均 Macro-F1 高 {df:.3f} 个百分点，未作显著性检验。'
    docs+='主汇总及其总体/每类 CSV 已纳入 CUFIT；HQ 未补跑，仅报告 Full。完整报告见 [cufit_comparison_results/COMPARISON_REPORT.md](cufit_comparison_results/COMPARISON_REPORT.md)。\n\n'
    docs+='训练及权重源目录：`../CUFIT0613/experiment_full_vs_ours_20261022/`。原训练代码、数据集、历史模型指标和 `summary_full/hq/compare.csv` 的早期原始记录保留；本次只向 3-seed 分析增加结果。\n'
    updated={'analysis_3seed_summary.md':text,
             'analysis_3seed_overall.csv':append_csv(oldtexts['analysis_3seed_overall.csv'],[overall]),
             'analysis_3seed_perclass.csv':append_csv(oldtexts['analysis_3seed_perclass.csv'],class_rows),
             'PROJECT_DOCS.md':docs}
    # Verify old CSV records remain identical, and only the intended rows were added.
    new_overall=list(csv.DictReader(io.StringIO(updated['analysis_3seed_overall.csv'])))
    new_classes=list(csv.DictReader(io.StringIO(updated['analysis_3seed_perclass.csv'])))
    assert new_overall[:-1]==legacy_overall and len(new_overall)==len(legacy_overall)+1
    assert new_classes[:-3]==legacy_classes and len(new_classes)==len(legacy_classes)+3
    assert new_overall[-1]==overall and new_classes[-3:]==class_rows
    assert text.count('## 表9')==1
    assert len(re.findall(r'^## 表[0-9]',text,re.M))==9
    for seed in (42,43,44):
        assert len([r for r in per_seed if r['seed']==seed])==1
    DEST.mkdir(exist_ok=True)
    assert not (DEST/'INTEGRATION_AUDIT.json').exists(), 'Existing integration audit'
    stamp=datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    backup=DEST/'integration_backups'/stamp; backup.mkdir(parents=True)
    for name in targets: shutil.copy2(REPO/name,backup/name)
    for p in (EXP/'results').iterdir():
        if p.is_file(): shutil.copy2(p,DEST/p.name)
    for name in ['audit.json','references.json','execution_files_sha256.json','EXPERIMENT_PLAN.md']:
        shutil.copy2(EXP/name,DEST/name)
    pred_dir=DEST/'predictions';pred_dir.mkdir(exist_ok=True)
    copies=[]
    for r in result['per_seed']:
        label={'CUFIT + Full':'cufit_full','HQ + Ours':'ours_hq',
               'RVC + ResNet34 (reference)':'rvc_resnet34','RC + ResNet34 (reference)':'rc_resnet34'}[r['method']]
        src=Path(r['predictions']);dst=pred_dir/f'{label}_seed{r["seed"]}.csv'
        assert sha(src)==r['sha256'];shutil.copy2(src,dst)
        copies.append(dict(method=r['method'],seed=r['seed'],source=str(src),archive=str(dst.relative_to(DEST)),sha256=sha(dst)))
    dump(DEST/'PREDICTION_SOURCES.json',copies)
    with (DEST/'CUFIT_FULL_SEED_METRICS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per_seed[0]));w.writeheader();w.writerows(per_seed)
    # Publish with same-directory atomic replacement; retain exact original backups.
    changes=[]
    for name,newtext in updated.items():
        assert (REPO/name).read_bytes()==originals[name],f'Concurrent edit: {name}'
        p=REPO/name;tmp=p.with_name(p.name+'.cufit.tmp')
        with tmp.open('w',encoding='utf-8',newline='') as f:f.write(newtext)
        os.chmod(tmp,p.stat().st_mode & 0o777);tmp.replace(p)
        assert p.read_bytes()==newtext.encode('utf-8')
        assert p.stat().st_uid==os.getuid()
        changes.append(dict(path=str(p),before_sha256=hashlib.sha256(originals[name]).hexdigest(),after_sha256=sha(p),backup=str(backup/name)))
    audit=dict(integrated_at=datetime.now().astimezone().isoformat(),experiment=str(EXP),
               source_complete=True,cufit_seeds=[42,43,44],cufit_dataset='full',
               legacy_overall_rows_preserved=len(legacy_overall),legacy_perclass_rows_preserved=len(legacy_classes),
               new_overall_row=overall,new_perclass_rows=class_rows,changes=changes,
               prediction_archives=copies,script_sha256=sha(Path(__file__)),
               delta_ours_minus_cufit_pp=dict(macro_f1=df,accuracy=da,macro_auroc=du))
    dump(DEST/'INTEGRATION_AUDIT.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
