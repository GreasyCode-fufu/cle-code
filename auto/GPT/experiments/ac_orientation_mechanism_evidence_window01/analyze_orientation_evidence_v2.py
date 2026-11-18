"""Frozen probability-only validation diagnostics, no network execution or test-array reads."""
from pathlib import Path
import csv,hashlib,json,os,sys
from fractions import Fraction
import numpy as np
S=Path(__file__).resolve().parent;R=S.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def verify():
    revision=read(S/'analysis_revision_v2.json');assert revision['original_config_sha256']==sha(S/'config.json') and all(sha(S/n)==h for n,h in revision['sources'].items());c=read(S/'config.json');assert all(sha(R/p)==h for p,h in c['locked'].items());assert all(sha(S/n)==h for n,h in c['sources'].items());return c
def cm(pred,y,weight=None):
    return np.bincount(y*3+pred,weights=weight,minlength=9).reshape(3,3)
def f1(counts):
    tp=np.diagonal(counts,axis1=-2,axis2=-1);den=counts.sum(-2)+counts.sum(-1)
    return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0).mean(-1)
def fraction_f1(counts):
    counts=np.asarray(counts,dtype=np.int64)
    return float(sum((Fraction(2*int(counts[k,k]),int(counts[:,k].sum()+counts[k,:].sum())) if counts[:,k].sum()+counts[k,:].sum() else Fraction(0)) for k in range(3))/3)
def metrics(p,y,g):
    pred=p.argmax(1);matrix=cm(pred,y);groups=np.unique(g)
    return dict(macro_f1=float(f1(matrix)),accuracy=float(np.mean(pred==y)),malignant_recall=float(np.mean(pred[y==1]==1)),patient_equal_accuracy=float(np.mean([np.mean(pred[g==a]==y[g==a]) for a in groups])),confusion_matrix=matrix.astype(int).tolist())
def predictions(p):
    rows=list(csv.DictReader(p.open()));prefix='' if all(f'p{k}' in rows[0] for k in range(3)) else 'mean_';assert all(f'{prefix}p{k}' in rows[0] for k in range(3));return rows,np.array([[float(r[f'{prefix}p{k}']) for k in range(3)] for r in rows])
def bootstrap(preds_a,preds_b,y,g):
    groups=np.unique(g);G=len(groups);assert G==39
    def group_confusions(preds):return np.stack([[cm(pred[g==a],y[g==a]) for a in groups] for pred in preds])
    a,b=group_confusions(preds_a),group_confusions(preds_b)
    rng=np.random.default_rng(20261026);counts=rng.multinomial(G,np.ones(G)/G,size=20000)
    delta=(f1(np.einsum('rg,sgij->rsij',counts,a))-f1(np.einsum('rg,sgij->rsij',counts,b))).mean(1)
    return dict(patients=G,images=len(y),replicates=20000,seed=20261026,ci95=np.quantile(delta,[.025,.975]).tolist(),selection_bias_corrected=False)
def precheck():
    y=np.array([0,1,0,2,1,2]);g=np.array(['a','a','b','b','c','c']);pred=np.array([0,0,2,2,1,1]);multiplicity=np.array([2,0,1])
    groups=np.unique(g);mat=sum(multiplicity[k]*cm(pred[g==a],y[g==a]) for k,a in enumerate(groups))
    indices=np.concatenate([np.flatnonzero(g==a) for k,a in enumerate(groups) for _ in range(multiplicity[k])])
    assert np.array_equal(mat,cm(pred[indices],y[indices])) and abs(f1(mat)-fraction_f1(mat))<1e-15
    save(S/'numeric_precheck.json',dict(passed=True,mixed_label_patient_confusion_oracle=True,fraction_f1=True,training_updates=0,network_forwards=0))
def main():
    cfg=verify();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';os.nice(10);precheck()
    rec=read(S/'val_manifest.json');y=np.array([r['label'] for r in rec]);g=np.array([r['group_id'] for r in rec]);ids=[r['index'] for r in rec]
    assert len(ids)==len(set(ids))==427 and len(set(g))==39;summary={}
    for family in cfg['families']:
        units=[];c4preds=[];originalpreds=[]
        for seed in cfg['seeds']:
            root=S/'inputs'/family/f's{seed}';views=np.load(root/'val_views.npy');rows,p=predictions(root/'val_predictions.csv');base,bp=predictions(root/'original_validation_predictions.csv')
            assert views.shape==(4,427,3) and np.isfinite(views).all() and np.allclose(views.sum(-1),1,atol=1e-6)
            for rr in [rows,base]:assert [int(r['index']) for r in rr]==ids and [int(r['label']) for r in rr]==y.tolist()
            assert np.max(np.abs(views.mean(0)-p))<1e-7 and np.max(np.abs(views[0]-bp))<1e-6
            v=views.astype(float);entropy=lambda z:-(z*np.log(np.clip(z,1e-300,1))).sum(-1)
            js=entropy(v.mean(0))-entropy(v).mean(0);assert js.min()>-1e-12 and js.max()<np.log(3)+1e-12
            predicted=views.argmax(-1);disagree=(predicted!=predicted[0:1]).any(0);old=predicted[0];new=views.mean(0).argmax(1);wrong=old!=y;rescue=wrong&(new==y);harm=(~wrong)&(new!=y)
            assert abs(np.mean(new==y)-np.mean(old==y)-(rescue.sum()-harm.sum())/len(y))<1e-14
            byclass={str(k):dict(images=int((y==k).sum()),rescued=int((rescue&(y==k)).sum()),harmed=int((harm&(y==k)).sum())) for k in range(3)}
            perangle=[metrics(views[k],y,g) for k in range(4)];average=metrics(views.mean(0),y,g)
            for z in [*perangle,average]:assert abs(z['macro_f1']-fraction_f1(z['confusion_matrix']))<1e-14
            unit=dict(seed=seed,angle_metrics=perangle,c4_metrics=average,argmax_disagreement_images=int(disagree.sum()),argmax_disagreement_rate=float(disagree.mean()),patient_equal_disagreement=float(np.mean([disagree[g==a].mean() for a in np.unique(g)])),mean_js_nats=float(js.mean()),mean_js_when_original_wrong=float(js[wrong].mean()) if wrong.any() else None,mean_js_when_original_correct=float(js[~wrong].mean()) if (~wrong).any() else None,rescued_images=int(rescue.sum()),harmed_images=int(harm.sum()),by_class=byclass,
                patient_equal_accuracy_delta=float(np.mean([np.mean(rescue[g==a])-np.mean(harm[g==a]) for a in np.unique(g)])))
            image_rows=[dict(index=i,group_id=rec[k]['group_id'],label=int(y[k]),original_pred=int(old[k]),c4_pred=int(new[k]),argmax_disagrees=int(disagree[k]),js_nats=float(js[k]),rescued=int(rescue[k]),harmed=int(harm[k])) for k,i in enumerate(ids)]
            dest=S/'diagnostics'/family/f's{seed}.csv';dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(image_rows[0]));w.writeheader();w.writerows(image_rows)
            units.append(unit);c4preds.append(new);originalpreds.append(old)
        delta=np.array([u['c4_metrics']['macro_f1']-u['angle_metrics'][0]['macro_f1'] for u in units])
        summary[family]=dict(units=units,original_macro_f1=float(np.mean([u['angle_metrics'][0]['macro_f1'] for u in units])),c4_macro_f1=float(np.mean([u['c4_metrics']['macro_f1'] for u in units])),paired_seed_deltas=delta.tolist(),mean_delta=float(delta.mean()),conditional_patient_bootstrap=bootstrap(c4preds,originalpreds,y,g),mean_rescued_images=float(np.mean([u['rescued_images'] for u in units])),mean_harmed_images=float(np.mean([u['harmed_images'] for u in units])),mean_disagreement_images=float(np.mean([u['argmax_disagreement_images'] for u in units])))
    save(S/'orientation_evidence_results.json',dict(analysis_revision=2,revision_config_sha256=sha(S/'analysis_revision_v2.json'),summary=summary,val_images=427,val_patients=39,test_prediction_arrays_read=0,network_forwards=0,training_updates=0,independent_confirmation=False,existing_18_frozen_jobs_changed=False))
    lines=['# 方向机制val诊断：现有概率证据','','仅使用既有427张val、39患者、三seed的四角度概率，不运行网络或访问test预测数组。不是新的独立验证。','', '|骨干|原0度F1均值|C4均值|差（百分点）|39患者条件95%区间（百分点）|每seed平均纠正/损伤数|','|---|---:|---:|---:|---|---:|']
    for family,r in summary.items():
        lo,hi=r['conditional_patient_bootstrap']['ci95'];lines.append(f'|{family}|{100*r["original_macro_f1"]:.4f}%|{100*r["c4_macro_f1"]:.4f}%|{100*r["mean_delta"]:.4f}|[{100*lo:.4f}, {100*hi:.4f}]|{r["mean_rescued_images"]:.3f}/{r["mean_harmed_images"]:.3f}|')
    lines+=['','所有单角、逐seed指标和混淆矩阵、分歧、JS、纠正/损伤及类别记录见JSON/diagnostics。没有选择最优角度或组合，没有拟合阈值/权重。','', '方向分歧并不保证平均预测能纠错：必须同时记录新增错误。已有C4开发test改善不能替代学习型方向机制的重训与对照。已冻结18项实验保持不变，待实际结果判断方向矩是否优于普通旋转增强；不据此把成熟C4包装成网络创新。','', '原模型曾用val选epoch和B0，工作Swin还按test反馈选择；条件区间不校正选择偏差。历史test及A–C构建标签依赖如前披露，等级不是医生标签正确性证明。']
    (S/'ORIENTATION_MECHANISM_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
