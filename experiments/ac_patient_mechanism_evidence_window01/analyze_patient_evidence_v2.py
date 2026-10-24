"""Class-conditioned patient variance and two fixed train-fit prototype diagnostics."""
from pathlib import Path
import collections,csv,json,sys
import numpy as np
S=Path(__file__).resolve().parent;R=S.parents[1];sys.path.insert(0,str(S))
import patient_evidence as e
sys.path.insert(0,str(R/'experiments/ac_test_feedback_window01'))
from audit_test_feedback_v2 import compute_metrics

def conditional_bootstrap(a,b,y,groups):
    # Independent confusion accumulation; no test-only metadata constants.
    unique=sorted(set(groups));count=len(unique);assert count==39 and len(y)==427
    group_index=np.array([unique.index(group) for group in groups]);rng=np.random.default_rng(20261024)
    draws=rng.multinomial(count,np.full(count,1/count),size=20000);scores=[]
    assert np.all(draws.sum(1)==count)
    for collection in [a,b]:
        matrices=np.zeros((len(collection),count,3,3),dtype=np.int64)
        for seed,p in enumerate(collection):np.add.at(matrices[seed],(group_index,y,p.argmax(1)),1)
        sampled=np.einsum('dg,sgij->dsij',draws,matrices)
        tp=np.diagonal(sampled,axis1=-2,axis2=-1);den=sampled.sum(-1)+sampled.sum(-2)
        scores.append(np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0).mean((-1,-2)))
    delta=scores[0]-scores[1]
    return dict(ci95=np.percentile(delta,[2.5,97.5]).tolist(),draws=20000,patients=count,images=len(y),paired_seeds=3,
        patient_draws_shared_across_seeds=True,draw_group_count_verified=True,
        scope='Conditional val interval on39 patients; repeated development and prior W0 test selection remain.')

def prototypes(features,y,groups):
    image=[];patient=[]
    for label in range(3):
        subset=y==label;assert subset.any();image.append(features[subset].mean(0))
        # A patient may have several image labels. Never average a patient across disease classes.
        patient.append(np.mean([features[subset&(groups==group)].mean(0) for group in sorted(set(groups[subset]))],0))
    return np.array(image),np.array(patient)

def probability(features,centers):
    distances=((features[:,None,:]-centers[None,:,:])**2).sum(-1);logits=-distances;logits-=logits.max(1,keepdims=True)
    exp=np.exp(logits);return exp/exp.sum(1,keepdims=True)

def variance(features,y,groups):
    x=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12);rows=[];totals=np.zeros(3)
    for label in range(3):
        subset=y==label;mu=x[subset].mean(0);within=between=0.;total=float(((x[subset]-mu)**2).sum())
        for group in sorted(set(groups[subset])):
            pos=subset&(groups==group);center=x[pos].mean(0);within+=float(((x[pos]-center)**2).sum());between+=int(pos.sum())*float(((center-mu)**2).sum())
        assert abs(total-within-between)<max(1e-9,1e-9*total)
        rows.append(dict(label=label,images=int(subset.sum()),patient_classes=len(set(groups[subset])),within_patient_class_sse=within,
            between_patient_within_class_sse=between,within_class_total_sse=total,between_fraction=between/total if total else 0.))
        totals+=np.array([within,between,total])
    return dict(classes=rows,aggregate_between_fraction=float(totals[1]/totals[2]),decomposition_verified=True,
        limitation='Descriptive association in a fitted representation; does not establish patient style as a cause or independent out-of-patient performance.')

def group_description(rows):
    counts=collections.Counter(r['group_id'] for r in rows);labels=collections.defaultdict(set)
    for r in rows:labels[r['group_id']].add(r['label'])
    return dict(images=len(rows),patient_groups=len(counts),min_images=min(counts.values()),max_images=max(counts.values()),
        top10_images=sum(sorted(counts.values(),reverse=True)[:10]),image_weight_concentration_equivalent_groups=len(rows)**2/sum(n*n for n in counts.values()),
        mixed_label_groups=sum(len(v)>1 for v in labels.values()),
        class_images=dict(collections.Counter(r['label'] for r in rows)),class_patient_groups={str(k):len({r['group_id'] for r in rows if r['label']==k}) for k in range(3)},
        note='Equivalent group count is inverse image-weight concentration, not an independent-sample-size estimate.')

def main():
    import torch
    cfg=e.verify();revision=e.READ(S/'analysis_config_v2.json');assert revision['base_config_sha256']==e.sha(S/'config.json') and revision['analysis_source_sha256']==e.sha(__file__);records={r['index']:r for r in e.READ(S/'manifest.json')};ids=cfg['indices']['train']+cfg['indices']['val'];n=2077
    y=np.array([records[i]['label'] for i in ids]);g=np.array([records[i]['group_id'] for i in ids]);summaries=[];ps=collections.defaultdict(list)
    for unit in cfg['units']:
        seed=unit['seed'];proof=e.READ(S/f'features_s{seed}.json');assert proof['test']==proof['network_updates']==0 and proof['state_unchanged']
        assert all(e.sha(S/p)==h for p,h in proof['files'].items());dest=S/'features'/f's{seed}'
        x=np.load(dest/'features.npy').astype(float);logits=np.load(dest/'logits.npy').astype(float)
        assert np.load(dest/'indices.npy').tolist()==ids and np.load(dest/'labels.npy').tolist()==y.tolist() and x.shape==(2504,768)
        assert np.isfinite(x).all() and np.isfinite(logits).all()
        assert e.sha(R/unit['checkpoint'])==unit['checkpoint_sha256'];weights=torch.load(R/unit['checkpoint'],map_location='cpu',weights_only=True)
        reconstructed=x@weights['head.weight'].numpy().astype(float).T+weights['head.bias'].numpy().astype(float)
        head_error=float(np.abs(reconstructed-logits).max());assert head_error<1e-4
        image,patient=prototypes(x[:n],y[:n],g[:n]);independent=[]
        for label in range(3):
            subset=y[:n]==label;gs=g[:n][subset];w=np.array([1/np.sum(gs==group) for group in gs]);independent.append(np.average(x[:n][subset],axis=0,weights=w))
        assert np.max(np.abs(patient-np.array(independent)))<1e-10
        val_logits=logits[n:]-logits[n:].max(1,keepdims=True);head=np.exp(val_logits);head/=head.sum(1,keepdims=True)
        pimage=probability(x[n:],image);ppatient=probability(x[n:],patient)
        probs=dict(frozen_linear_head_cpu=head,image_mean_euclidean=pimage,patient_class_mean_euclidean=ppatient)
        source=R/unit['complete'];gpu=list(csv.DictReader((source.parent/'validation_predictions.csv').open()));gpu_p=np.array([[float(row[f'p{k}']) for k in range(3)] for row in gpu])
        assert [int(row['index']) for row in gpu]==cfg['indices']['val']
        agreement=float(np.mean(head.argmax(1)==gpu_p.argmax(1)));assert agreement>=.99,'Unexpected CPU/GPU disagreement: inspect input and precision before interpreting evidence'
        metrics={name:compute_metrics(p,y[n:],g[n:]) for name,p in probs.items()}
        for name,p in probs.items():ps[name].append(p)
        summary=dict(seed=seed,checkpoint_sha256=unit['checkpoint_sha256'],metrics=metrics,variance=variance(x[:n],y[:n],g[:n]),
            independent_linear_head_max_logit_error=head_error,prototype_inverse_count_oracle=True,
            cpu_gpu_val_prediction_agreement=agreement,cpu_gpu_max_probability_difference=float(np.abs(head-gpu_p).max()),
            prototype_shift_norm_by_class=np.linalg.norm(patient-image,axis=1).tolist(),train_fit_count=n,val_evaluation_count=427,test_count=0)
        summaries.append(summary)
        data=[]
        for pos,i in enumerate(cfg['indices']['val']):
            row=records[i];data.append(dict(index=i,group_id=row['group_id'],label=row['label'],**{name:int(p[pos].argmax()) for name,p in probs.items()}))
        e.core.save_csv(S/f'val_diagnostic_predictions_s{seed}.csv',data)
    stats={name:dict(seed_macro_f1=[u['metrics'][name]['macro_f1'] for u in summaries],
        **{key:float(np.mean([u['metrics'][name][key] for u in summaries])) for key in ['macro_f1','accuracy','malignant_recall','patient_equal_accuracy']}) for name in ps}
    comparisons={}
    for a,b in [('patient_class_mean_euclidean','image_mean_euclidean'),('patient_class_mean_euclidean','frozen_linear_head_cpu'),('image_mean_euclidean','frozen_linear_head_cpu')]:
        differences=np.array(stats[a]['seed_macro_f1'])-np.array(stats[b]['seed_macro_f1'])
        comparisons[a+'_vs_'+b]=dict(paired_seed_deltas=differences.tolist(),mean_delta=float(differences.mean()),
            conditional_val_patient_bootstrap=conditional_bootstrap(ps[a],ps[b],y[n:],g[n:]))
    previous=e.READ(S/'analysis_v1/patient_evidence_results.json')
    for name,value in comparisons.items():assert np.allclose(value['conditional_val_patient_bootstrap']['ci95'],previous['comparisons'][name]['conditional_val_patient_bootstrap']['ci95'],atol=1e-12,rtol=0), 'Unexpected change beyond CI metadata correction'
    result=dict(analysis_revision=2,analysis_source_sha256=e.sha(__file__),analysis_config_sha256=e.sha(S/'analysis_config_v2.json'),prior_numeric_confidence_intervals_reproduced=True,created_at=e.core.timestamp(),reference=cfg['reference'],train_description=group_description([records[i] for i in cfg['indices']['train']]),
        val_description=group_description([records[i] for i in cfg['indices']['val']]),units=summaries,summary=stats,comparisons=comparisons,
        config_sha256=e.sha(S/'config.json'),extraction_pipeline_source_hashes=e.sources(),test_images=0,network_updates=0,independent_validation=False,
        limitation='Only train/val images used directly, but mature reference had prior adaptive test selection. Fitted encoder saw all train patients. Not OOF, not independent validation, not a new trained network contribution.')
    e.SAVE(S/'patient_evidence_results.json',result);e.SAVE(S/'independent_computational_audit.json',dict(passed=True,analysis_revision=2,seeds=cfg['seeds'],train=2077,val=427,val_patients=39,bootstrap_draw_group_count_verified=True,prior_CI_numeric_values_reproduced=True,test=0,
        fixed_sample_identity=True,source_checkpoint_feature_hashes=True,unchanged_model_state=True,linear_head_reconstructed=True,
        mixed_label_patient_conditioning_verified=True,variance_decomposition_verified=True,prototype_inverse_count_oracle=True,independent_validation=False))
    lines=['# 患者差异机制证据：仅train/val，CPU分析','',
        '本阶段不更新网络、不改变H1训练，不读test图像。W0先前按test反馈选择，故此分析并非完全没有test筛选依赖。已拟合encoder见过全部train患者；这里没有患者OOF或独立验证。原型只采用固定train类均值和距离公式作诊断，不是原始Prototypical Networks完整episodic训练，也不声称创新。','',
        '修订记录：v1复用了test helper，抽样本身使用实际39个val患者，但返回元数据将患者数写死为41。v2用独立混淆计数重建39患者bootstrap，逐项确认原区间数值不变；原v1源码/报告/审计另存，不重新提取特征、不训练模型。','', '## 清单证据','']
    for split in ['train','val']:
        d=result[split+'_description'];lines.append(f'{split}：{d["images"]}图/{d["patient_groups"]}患者；每患者{d["min_images"]}–{d["max_images"]}图，最多10患者占{d["top10_images"]}图；{d["mixed_label_groups"]}患者含多个类别。图像均匀权重的患者集中度等效数{d["image_weight_concentration_equivalent_groups"]:.3f}，只是权重集中度，不是独立样本量。')
    lines+=['','因此不能把每患者当一个固定疾病标签，也不能将其全部图像直接平均后赋单一标签。这里只在每个类别内部构造患者均值，再按患者等权汇总；图像清单、标签、A–C等级完全保留。','',
        '|固定诊断|seed45 val F1|seed46|seed47|均值|患者等权accuracy|','|---|---:|---:|---:|---:|---:|']
    for name,m in stats.items():lines.append('|'+name+'|'+'|'.join(f'{100*v:.4f}%' for v in m['seed_macro_f1']+[m['macro_f1'],m['patient_equal_accuracy']])+'|')
    lines+=['','各类别同类方差中患者间部分占比、逐seed混淆矩阵、CPU/GPU数值差异与条件患者bootstrap见JSON。若患者等权原型没有稳定收益，不能把患者原型机制写成有效；此证据只用于决定下一机制，不自动启动近似组合搜索。','',
        '## 成熟方法参照','',
        '均值原型距离是已有方法：[Prototypical Networks，NIPS2017](https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html)。这里仅借用固定embedding下的类均值距离，不复制其完整学习流程。',
        '有监督同类聚合已有[SupCon，NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)；群组稳健优化已有[Group DRO官方代码](https://github.com/kohpangwei/group_DRO)。未来若采用相关机制，须保留忠实参照、预算匹配与消融；不能将成熟思想重新命名为已证明新颖。']
    (S/'PATIENT_EVIDENCE_REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');e.commit('evidence: independently reconstruct all fixed prototype diagnostics and retain patient variance findings')
    print(json.dumps(dict(summary=stats,comparisons=comparisons),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
