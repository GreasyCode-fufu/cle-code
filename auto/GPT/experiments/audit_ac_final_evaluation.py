"""Independent reproduction of the single frozen AC final evaluation."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_ac_final_evaluation as study
from pretrained_representations import state_sha as source_state_sha
from predict_ac_model import load_model,predict_paths
core=study.core;s=study.STUDY;cfg=study.verify()
study.first.verify();study.second.verify();study.third.verify()
records=core.read_json(s/'manifest.json');jobs=core.read_json(s/'final_jobs.json')
assert len(jobs)==39 and all(study.done(j) for j in jobs)
cohort=core.read_json(study.COHORT/'dataset_config.json');score=cohort['indices']['test']
groups=sorted({records[i]['group_id'] for i in score});assert len(groups)==41 and len(score)==576
group_index={g:i for i,g in enumerate(groups)}
def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def signature(job, config):
    return hashlib.sha256((json.dumps(job, sort_keys=True) + core.digest(config) + core.digest(core.__file__)).encode()).hexdigest()


def state_sha(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def validate_predictions(path, indices):
    rr = rows(path)
    assert [int(row['index']) for row in rr] == indices
    assert len(rr) == len(set(indices)) == 576
    p = np.array([[float(row[f'p{k}']) for k in range(3)] for row in rr])
    y = np.array([int(row['label']) for row in rr])
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=2e-6)
    assert np.array_equal(p.argmax(1), [int(row['pred']) for row in rr])
    for row in rr:
        rec = records[int(row['index'])]
        assert row['sample_id'] == rec['sample_id'] and row['group_id'] == rec['group_id'] and int(row['label']) == rec['label']
    matrix = np.zeros((41, 3, 3), dtype=np.float64)
    np.add.at(matrix, ([group_index[row['group_id']] for row in rr], y, p.argmax(1)), 1)
    return p, y, matrix


def metric_from_cm(matrix):
    den = matrix.sum(-1) + matrix.sum(-2)
    return np.mean(np.divide(2 * np.diagonal(matrix, axis1=-2, axis2=-1), den,
                             out=np.zeros_like(den), where=den != 0), axis=-1)



predictions={};models=[];source_copies={};maxdiff=0.
for job in jobs:
    old=job['source_job'];folder=ROOT/job['source_folder'];source=core.read_json(folder/'runs'/old['id']/'complete.json')
    expected=hashlib.sha256((json.dumps(old,sort_keys=True)+core.digest(folder/'config.json')+core.digest(core.__file__)).encode()).hexdigest()
    assert expected==job['source_signature']==source['signature']
    assert source['checkpoint']==job['checkpoint'] and source['checkpoint_sha256']==job['checkpoint_sha256']==core.digest(ROOT/job['checkpoint'])
    assert old['train']==cohort['indices']['train'] and old['select']==old['score']==cohort['indices']['val']
    assert not {records[i]['group_id'] for i in score}&{records[i]['group_id'] for i in old['train']+old['select']}
    assert {records[i]['split'] for i in job['score']}=={'test'} and job['score']==score
    out=s/'runs'/job['id'];result=core.read_json(out/'test_complete.json')
    sig=hashlib.sha256((json.dumps(job,sort_keys=True)+core.digest(s/'config.json')).encode()).hexdigest()
    assert result['signature']==sig and result['checkpoint_sha256']==job['checkpoint_sha256']
    assert result['prediction_sha256']==core.digest(out/'test_predictions.csv') and result['state_unchanged']
    saved,y,matrix=validate_predictions(out/'test_predictions.csv',score);metric=core.metrics(saved,y)
    assert abs(metric['macro_f1']-result['macro_f1'])<1e-12 and metric['confusion_matrix']==result['confusion_matrix']
    model=study.model_for(job);state=study.torch_load(ROOT/job['checkpoint'])
    assert all(torch.isfinite(v).all() for v in state.values());model.load_state_dict(state,strict=True);model.eval()
    parameters=sum(p.numel() for p in model.parameters())
    before=state_sha(model);assert source_state_sha(model)==result['state_sha256']
    p,yr,idx=core.infer(model,core.loader(score,False,seed=job['seed']))
    difference=float(np.max(np.abs(p-saved)));maxdiff=max(maxdiff,difference)
    assert difference<2e-6 and idx==score and np.array_equal(yr,y) and np.array_equal(p.argmax(1),saved.argmax(1))
    assert state_sha(model)==before
    key=(job['family'],job['seed']);assert key not in predictions
    predictions[key]=matrix;source_copies[job['id']]=saved
    cm=np.asarray(metric['confusion_matrix'])
    models.append(dict(job=job['id'],family=job['family'],seed=job['seed'],source_kind=job['source_kind'],
        checkpoint_sha256=job['checkpoint_sha256'],parameters=parameters,validation_f1=job['validation_f1'],
        test_macro_f1=metric['macro_f1'],test_accuracy=metric['accuracy'],malignant_recall=float(cm[1,1]/cm[1].sum()),
        patient_equal_accuracy=float(np.mean(np.trace(matrix,axis1=-2,axis2=-1)/matrix.sum((1,2)))),
        confusion_matrix=metric['confusion_matrix'],probability_max_difference=difference,inference_state_unchanged=True))
    assert (ROOT/job['checkpoint']).owner()=='login' and all(f.owner()=='login' for f in out.rglob('*') if f.is_file())
    print('audited',job['id'],flush=True);del model,state;torch.cuda.empty_cache()

weights=np.random.default_rng(cfg['bootstrap_rng']).multinomial(41,np.ones(41)/41,size=cfg['bootstrap_draws'])
comparisons=[]
for control in cfg['references']:
    ds=[];boot=[]
    for seed in cfg['primary_seeds']:
        candidate,reference=predictions['dino_full',seed],predictions[control,seed]
        ds.append(float(metric_from_cm(candidate.sum(0))-metric_from_cm(reference.sum(0))))
        boot.append(metric_from_cm(np.einsum('bg,gij->bij',weights,candidate))-metric_from_cm(np.einsum('bg,gij->bij',weights,reference)))
    distribution=np.mean(boot,axis=0);family=np.percentile(distribution,cfg['family_percentiles'])
    comparisons.append(dict(candidate='dino_full',reference=control,seed_deltas=ds,mean_delta=float(np.mean(ds)),
        patient95=np.percentile(distribution,[2.5,97.5]).tolist(),family_interval=family.tolist(),family_percentiles=cfg['family_percentiles'],
        fixed_cohort_pair_supported=bool(min(ds)>0 and np.mean(ds)>=cfg['minimum_delta'] and family[0]>0)))
whole=all(r['fixed_cohort_pair_supported'] for r in comparisons)
truth,predicted=np.indices((3,3)).reshape(2,-1)
for matrix in predictions.values():
    for draw in weights[:8]:
        cm=np.einsum('g,gij->ij',draw,matrix)
        assert abs(f1_score(truth,predicted,sample_weight=cm.ravel(),labels=[0,1,2],average='macro',zero_division=0)-float(metric_from_cm(cm)))<1e-12
summary=[]
for family in sorted({r['family'] for r in models}):
    rr=[r for r in models if r['family']==family]
    summary.append(dict(family=family,seeds=[r['seed'] for r in rr],n_models=len(rr),seed_test_f1=[r['test_macro_f1'] for r in rr],
        mean_validation_f1=float(np.mean([r['validation_f1'] for r in rr])),mean_test_f1=float(np.mean([r['test_macro_f1'] for r in rr])),
        test_seed_sd=float(np.std([r['test_macro_f1'] for r in rr],ddof=1)),mean_malignant_recall=float(np.mean([r['malignant_recall'] for r in rr])),
        mean_patient_accuracy=float(np.mean([r['patient_equal_accuracy'] for r in rr])),parameters=rr[0]['parameters']))

deployment=core.read_json(s/'deployment_manifest.json');api_results=[]
paths=[ROOT/records[i]['path'] for i in score]
for name,item in deployment['models'].items():
    assert item['seed']==45 and core.digest(ROOT/item['checkpoint'])==item['checkpoint_sha256']
    model,_=load_model(s/'deployment_manifest.json',name);before=state_sha(model)
    output=predict_paths(model,paths);probabilities=np.asarray([r['probabilities'] for r in output])
    original=source_copies[item['final_job_id']]
    difference=float(np.max(np.abs(probabilities-original)))
    assert difference<2e-6 and np.array_equal(probabilities.argmax(1),original.argmax(1)) and state_sha(model)==before
    api_results.append(dict(model=name,images=576,preprocessing='Original-image RGB bilinear256/center224/ImageNet',
        probability_max_difference=difference,checkpoint_sha256=item['checkpoint_sha256'],state_unchanged=True))
    del model;torch.cuda.empty_cache()
    cpu_model,_=load_model(s/'deployment_manifest.json',name,device='cpu')
    cpu_output=predict_paths(cpu_model,[ROOT/records[cohort['indices']['val'][0]]['path']],device='cpu')
    cp=np.asarray(cpu_output[0]['probabilities']);assert np.isfinite(cp).all() and abs(cp.sum()-1)<2e-6
    api_results[-1]['cpu_single_validation_image_passed']=True
    del cpu_model
core.save_json(s/'deployment_validation.json',dict(passed=True,results=api_results,selection_unchanged=True))
core.save_json(s/'final_results.json',models);core.save_json(s/'final_summary.json',summary)
core.save_json(s/'paired_comparisons.json',dict(comparisons=comparisons,fixed_cohort_system_supported=whole))
core.save_json(s/'independent_audit.json',dict(passed=True,timestamp=core.timestamp(),models=39,test_images=576,test_groups=41,
    new_neural_training=0,all_checkpoint_signatures_hashes_verified=True,all_model_probabilities_reproduced=True,
    probability_max_difference=maxdiff,patients_disjoint_from_model_train_validation=True,deployment_api_full_test_verified=True,
    model_selection_unchanged=True,scope='One frozen conditional AC cohort evaluation; numerical repeat is audit only, not reselection.',
    checks=['39 unique frozen checkpoints from three completed rounds','all576 per-image labels/probabilities/argmax/metrics',
            'strict reload/finite params/frozen inference state/SHA/source signature','41patient shared bootstrap/three-family complete condition',
            'sklearn independent weighted-label F1','both standalone APIs exactly reproduce all test source predictions','login ownership']))
lines=['# 固定A–C数据集：最终统一评价','',
    '开发后冻结DINO完整微调为探索性质量优先候选，CNN为低计算参照；部署均固定seed45，未按test换模型/seed。R3验证完整门槛false仍保留。',
    '39个既有模型统一评价同576张ACtest/41患者组，无新训练、温度拟合、标签替换或重筛。不同家族3/6个模型，数量显式列出；主要比较只用配对45/46/47。','',
    '| 流程 | seed数 | 平均val F1 | 平均test F1 | 恶性召回 | 患者等权准确率 | 参数 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f'| {r["family"]} | {r["n_models"]} | {r["mean_validation_f1"]:.6f} | {r["mean_test_f1"]:.6f} | {r["mean_malignant_recall"]:.6f} | {r["mean_patient_accuracy"]:.6f} | {r["parameters"]} |')
lines+=['','固定最终DINO比较，41患者20000共享bootstrap；差值为百分点，普通95%和三比较家族区间。','',
    '| DINO相对 | 三seed差 | 平均差 | 普通95% | 家族区间 | 固定人群单项支持 |','|---|---|---:|---|---|---|']
for r in comparisons:lines.append(f'| {r["reference"]} | {[round(x*100,4) for x in r["seed_deltas"]]} | {100*r["mean_delta"]:+.4f} | {[round(x*100,4) for x in r["patient95"]]} | {[round(x*100,4) for x in r["family_interval"]]} | {r["fixed_cohort_pair_supported"]} |')
lines+=['',f'固定测试人群完整三项支持条件：{whole}。不回改R3验证四家族失败，不根据本表最高test值改候选或继续调参。',
    'CNN与DINO预训练、架构、维数及计算不同，约7.33G/12.25G计数FLOPs，不归因单一自监督或新结构；三轮没有确认自主新增结构贡献。',
    'test的A–C定义使用过其标签及内部交叉拟合，历史test也用过；筛选依赖模型一致性而非医生复核金标准，可能偏向CNN。结论限这个固定AC分布，不称原始完整人群或外部独立临床确认。',
    '患者区间条件于冻结模型/筛选清单，未覆盖所有选择、分级、交叉拟合依赖和训练/预训练不确定性。研究预测概率未临床校准。',
    '独立审计已重新加载39模型复现全部576概率，并核验两部署API从原图读取时对全部test的预测等价；原模型/标签/清单不变。数值重推是本次评价审计，不是新一轮试验。',
    '可用接口experiments/predict_ac_model.py，支持单图或目录；默认完整DINO，--kind cnn为低计算备选，--device cpu可在CPU使用。',
    '部署清单deployment_manifest.json锁服务器checkpoint路径/SHA。权重完整含主干，预测接口无需下载原预训练权重，不依赖等级或标签。默认及备选都固定seed45，所有seed原始结果另存。']
(s/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
assert all(p.owner()=='login' for p in s.rglob('*') if p.is_file())
subprocess.run([core.GIT,'-C',str(ROOT),'add','experiments/audit_ac_final_evaluation.py',str(s.relative_to(ROOT))],check=True)
subprocess.run([core.GIT,'-C',str(ROOT),'diff','--cached','--check'],check=True)
subprocess.run([core.GIT,'-C',str(ROOT),'commit','-m','results: independently audit all frozen AC test models and usable prediction interfaces'],check=True)
core.save_json(s/'status.json',dict(phase='final_audited_needs_delivery',updated_at=core.timestamp(),completed=39,models=39,fixed_cohort_system_supported=whole))
print(json.dumps(dict(passed=True,summary=summary,comparisons=comparisons,fixed_cohort_system_supported=whole,deployment_api=api_results)),flush=True)
