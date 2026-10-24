"""Independent AC backbone/update-recipe completion audit; never trains."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import math
import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_ac_backbone_recipe as study
from pretrained_representations import state_sha as source_state_sha
core=study.core;s=study.STUDY;cfg=study.verify()
cohort=core.read_json(study.COHORT/'dataset_config.json')
records=core.read_json(s/'manifest.json');jobs=core.read_json(s/'final_jobs.json')
assert len(jobs)==9 and all(study.done(j) for j in jobs)
assert core.read_json(study.prior.STUDY/'independent_audit.json')['passed']
assert core.digest(core.CACHE/'resized_uint8.npy')==core.read_json(core.CACHE/'meta.json')['cache_sha256']
groups=sorted({records[i]['group_id'] for i in cohort['indices']['val']});assert len(groups)==39
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
    assert len(rr) == len(set(indices)) == 427
    p = np.array([[float(row[f'p{k}']) for k in range(3)] for row in rr])
    y = np.array([int(row['label']) for row in rr])
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=2e-6)
    assert np.array_equal(p.argmax(1), [int(row['pred']) for row in rr])
    for row in rr:
        rec = records[int(row['index'])]
        assert row['sample_id'] == rec['sample_id'] and row['group_id'] == rec['group_id'] and int(row['label']) == rec['label']
    matrix = np.zeros((39, 3, 3), dtype=np.float64)
    np.add.at(matrix, ([group_index[row['group_id']] for row in rr], y, p.argmax(1)), 1)
    return p, y, matrix


def metric_from_cm(matrix):
    den = matrix.sum(-1) + matrix.sum(-2)
    return np.mean(np.divide(2 * np.diagonal(matrix, axis1=-2, axis2=-1), den,
                             out=np.zeros_like(den), where=den != 0), axis=-1)



reference_index=core.read_json(s/'reference_index.json');assert len(reference_index)==3
all_jobs=[]
for reference in reference_index:
    job=reference['source_job'];seed=job['seed'];original=study.prior.STUDY/'runs'/job['id']
    for name in ['complete.json','run_meta.json','train_log.csv','validation_predictions.csv','validation_complete.json']:
        assert (s/'references'/f's{seed}'/name).read_bytes()==(original/name).read_bytes()
    all_jobs.append((job,original,'cnn_original',study.prior.STUDY/'config.json'))
all_jobs += [(j,s/'runs'/j['id'],j['arm'],s/'config.json') for j in jobs]
budgets={b['arm']:b for b in core.read_json(s/'preflight_validation.json')['budgets']}
models=[];curves=[];predictions={};errors=collections.defaultdict(lambda:dict(wrong_votes=0,workflows=set()))
for job,out,arm,config in all_jobs:
    seed=job['seed'];assert seed in [45,46,47]
    assert job['train']==cohort['indices']['train'] and not job['sample_weights']
    assert job['select']==job['score']==cohort['indices']['val']
    assert len(job['train'])==2077 and len(job['score'])==427
    assert {records[i]['split'] for i in job['train']}=={'train'}
    assert {records[i]['split'] for i in job['score']}=={'val'}
    assert not {records[i]['group_id'] for i in job['train']} & {records[i]['group_id'] for i in job['score']}
    assert not set(job['train']+job['score']) & set(cohort['indices']['test'])
    comp=core.read_json(out/'complete.json');meta=core.read_json(out/'run_meta.json');result=core.read_json(out/'validation_complete.json')
    expected=signature(job,config)
    assert comp['signature']==meta['signature']==result['signature']==expected
    assert core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']
    assert comp['epochs_completed']==60 and comp['train_count']==2077 and comp['select_count']==427
    pretrained=cfg['resnet_pretrained_sha256'] if arm.startswith('cnn_') else cfg['dino_pretrained_sha256']
    assert comp['pretrained_sha256']==meta['pretrained_sha256']==pretrained
    log=rows(out/'train_log.csv');assert [int(r['epoch']) for r in log]==list(range(60))
    assert all(np.isfinite(float(v)) for row in log for v in row.values())
    best=int(np.argmax([float(r['val_macro_f1']) for r in log]));assert best==comp['best_epoch']
    assert abs(float(log[best]['val_macro_f1'])-comp['best_select_f1'])<1e-12
    for epoch,row in enumerate(log):
        factor=(1+math.cos(math.pi*(epoch+1)/60))/2
        if arm=='cnn_original':assert abs(float(row['lr'])-1e-4*factor)<1e-12
        else:
            first=1e-4 if arm=='dino_frozen' else 1e-5
            assert abs(float(row['lr_first'])-first*factor)<1e-12 and abs(float(row['lr_head'])-1e-4*factor)<1e-12
            assert int(row['amp_skipped_steps'])>=0
    saved,y,matrix=validate_predictions(out/'validation_predictions.csv',job['score']);metric=core.metrics(saved,y)
    assert abs(metric['macro_f1']-comp['best_select_f1'])<1e-12 and abs(metric['macro_f1']-result['macro_f1'])<1e-12
    assert metric['confusion_matrix']==result['confusion_matrix']
    if arm=='cnn_original':
        pcfg=core.read_json(study.prior.STUDY/'config.json')
        model=study.prior.model_for(seed,'plain',pcfg['control_width']);encoder=model.backbone
    else:
        assert result['checkpoint_sha256']==comp['checkpoint_sha256']
        assert result['prediction_sha256']==core.digest(out/'validation_predictions.csv')
        model=study.model_for(seed,arm);encoder=model.encoder
    initial={k:v.detach().cpu().clone() for k,v in encoder.state_dict().items() if not (arm=='cnn_original' and k.startswith('fc.'))}
    initial_bn={k:int(v) for k,v in model.state_dict().items() if k.endswith('num_batches_tracked')}
    state=torch.load(ROOT/comp['checkpoint'],map_location='cpu',weights_only=True)
    assert all(torch.isfinite(v).all() for v in state.values());model.load_state_dict(state,strict=True)
    parameters=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if arm=='cnn_original':assert parameters==trainable==21286211
    else:
        assert parameters==comp['parameters']==result['parameters']==budgets[arm]['parameters']
        assert trainable==comp['trainable_parameters']==result['trainable_parameters']==budgets[arm]['trainable_parameters']
        optimizer=study.transfer.optimizer_for(model)
        ids=[id(p) for group in optimizer.param_groups for p in group['params']]
        assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
        assert [g['lr'] for g in optimizer.param_groups]==([1e-4] if arm=='dino_frozen' else [1e-5,1e-4]);del optimizer
    assert len(initial_bn)==(36 if arm.startswith('cnn_') else 0)
    for name,value in initial_bn.items():assert int(state[name])==value+65*(best+1)
    squared=count=changed=0
    for name,original in initial.items():
        current=encoder.state_dict()[name].detach().cpu();changed+=int(not torch.equal(current,original))
        if current.is_floating_point():squared+=float((current.double()-original.double()).square().sum());count+=current.numel()
        if arm=='dino_frozen':assert torch.equal(current,original),name
    drift=(squared/count)**.5
    assert (changed==0)==(arm=='dino_frozen')
    if arm!='cnn_original':
        assert changed==result['source_tensors_changed'] and abs(drift-result['source_rms_drift'])<1e-12
    model.eval();before=state_sha(model)
    if arm!='cnn_original':assert source_state_sha(model)==result['inference_state_sha256'] and result['state_unchanged']
    p,yr,idx=core.infer(model,core.loader(job['score'],False,seed=seed))
    diff=float(np.max(np.abs(saved-p)))
    assert idx==job['score'] and np.array_equal(yr,y) and diff<2e-6 and np.array_equal(saved.argmax(1),p.argmax(1))
    assert state_sha(model)==before
    cm=np.array(metric['confusion_matrix']);predictions[arm,seed]=matrix
    models.append(dict(job=job['id'],arm=arm,seed=seed,checkpoint_sha256=comp['checkpoint_sha256'],signature=expected,
        macro_f1=metric['macro_f1'],accuracy=metric['accuracy'],malignant_recall=float(cm[1,1]/cm[1].sum()),
        patient_equal_accuracy=float(np.mean(np.trace(matrix,axis1=-2,axis2=-1)/matrix.sum((1,2)))),
        confusion_matrix=metric['confusion_matrix'],parameters=parameters,trainable_parameters=trainable,
        source_state_rms_drift=drift,source_tensors_changed=changed,bn_buffers_verified=len(initial_bn),
        inference_state_unchanged=True,probability_max_difference=diff))
    curves.append(dict(arm=arm,seed=seed,best_epoch=best,best_val_f1=metric['macro_f1'],final_val_f1=float(log[-1]['val_macro_f1']),
        final_train_accuracy=float(log[-1]['train_acc']),amp_skipped_steps=sum(int(r.get('amp_skipped_steps',0)) for r in log),history=log))
    for i,yi,pi in zip(idx,y,p.argmax(1)):
        if yi!=pi:errors[i]['wrong_votes']+=1;errors[i]['workflows'].add(arm)
    assert (ROOT/comp['checkpoint']).owner()=='login' and all(f.owner()=='login' for f in out.rglob('*') if f.is_file())
    print('audited',job['id'],flush=True)
    del model,encoder,state,initial;torch.cuda.empty_cache()

weights=np.random.default_rng(20261024).multinomial(39,np.ones(39)/39,size=20000)
pairs=[('cnn_low_lr','cnn_original'),('dino_full','cnn_original'),('dino_full','cnn_low_lr'),('dino_full','dino_frozen')]
comparisons=[]
for candidate,reference in pairs:
    ds=[];boot=[]
    for seed in cfg['seeds']:
        a,b=predictions[candidate,seed],predictions[reference,seed]
        ds.append(float(metric_from_cm(a.sum(0))-metric_from_cm(b.sum(0))))
        boot.append(metric_from_cm(np.einsum('bg,gij->bij',weights,a))-metric_from_cm(np.einsum('bg,gij->bij',weights,b)))
    distribution=np.mean(boot,axis=0);family=np.percentile(distribution,[.625,99.375])
    comparisons.append(dict(candidate=candidate,reference=reference,seed_deltas=ds,mean_delta=float(np.mean(ds)),
        patient95=np.percentile(distribution,[2.5,97.5]).tolist(),family_interval=family.tolist(),family_percentiles=[.625,99.375],
        pair_supported=bool(min(ds)>0 and np.mean(ds)>=cfg['minimum_delta'] and family[0]>0)))
gates=dict(cnn_update_supported=comparisons[0]['pair_supported'],dino_system_supported=all(r['pair_supported'] for r in comparisons[1:]))
truth,predicted=np.indices((3,3)).reshape(2,-1)
for matrix in predictions.values():
    for draw in weights[:8]:
        cm=np.einsum('g,gij->ij',draw,matrix)
        assert abs(f1_score(truth,predicted,sample_weight=cm.ravel(),labels=[0,1,2],average='macro',zero_division=0)-float(metric_from_cm(cm)))<1e-12
summary=[]
for arm in ['cnn_original']+study.ARMS:
    rr=[r for r in models if r['arm']==arm];assert len(rr)==3
    summary.append(dict(arm=arm,seed_f1=[r['macro_f1'] for r in rr],mean_f1=float(np.mean([r['macro_f1'] for r in rr])),
        seed_sd=float(np.std([r['macro_f1'] for r in rr],ddof=1)),mean_malignant_recall=float(np.mean([r['malignant_recall'] for r in rr])),
        mean_patient_accuracy=float(np.mean([r['patient_equal_accuracy'] for r in rr])),parameters=rr[0]['parameters'],trainable_parameters=rr[0]['trainable_parameters']))
resource_rows=rows(s/'resource_samples.csv');steady=[r for r in resource_rows[15:] if int(r['running'])>=6]
resource=dict(samples=len(resource_rows),six_job_samples=len(steady),mean_gpu_percent=float(np.mean([float(r['gpu_util']) for r in steady])),
    mean_cpu_percent=float(np.mean([float(r['cpu_percent']) for r in steady])),mean_iowait_percent=float(np.mean([float(r['cpu_iowait_percent']) for r in steady])))
error_rows=sorted([dict(index=i,sample_id=records[i]['sample_id'],group_id=records[i]['group_id'],label=records[i]['label'],
    wrong_votes=v['wrong_votes'],workflows=sorted(v['workflows'])) for i,v in errors.items()],key=lambda r:(-r['wrong_votes'],r['index']))
core.save_json(s/'independent_audit.json',dict(passed=True,timestamp=core.timestamp(),new_neural_models=9,reused_references=3,models=models,
    checks=['source/runtime/dual pretrained/reference signatures and SHA','60epochs/earliest-best/LR schedule/AMP log',
            'fixed2077train/427val/patient isolation/no test','all12 best model full-val predictions and inference state',
            'frozen DINO exact source; full source drift, finite params/budgets/optimizer coverage/CNN BN counts',
            '39patient four-family bootstrap/two complete questions and sklearn weighted-F1 crosscheck','login ownership']))
core.save_json(s/'round3_results.json',models);core.save_json(s/'round3_summary.json',summary)
core.save_json(s/'paired_comparisons.json',dict(comparisons=comparisons,**gates))
core.save_json(s/'training_curves.json',curves);core.save_json(s/'error_distribution.json',dict(val_errors_only=True,rows=error_rows))
core.save_json(s/'resource_summary.json',resource)
lines=['# A–C第三轮：主干与更新食谱结果','',
    '9新模型、3旧CNN参照；固定ACtrain2077/val427，同seed45/46/47，未评价test。','',
    '| 流程 | 三seed宏F1 | 平均F1 | 恶性召回 | 患者等权准确率 | 参数/可训练 |','|---|---|---:|---:|---:|---|']
for r in summary:lines.append(f'| {r["arm"]} | {r["seed_f1"]} | {r["mean_f1"]:.6f} | {r["mean_malignant_recall"]:.6f} | {r["mean_patient_accuracy"]:.6f} | {r["parameters"]}/{r["trainable_parameters"]} |')
lines+=['','39患者20000共享bootstrap；差值单位百分点。','',
    '| 比较 | 三seed差 | 均值差 | 普通95% | 四家族区间 | 单项支持 |','|---|---|---:|---|---|---|']
for r in comparisons:lines.append(f'| {r["candidate"]} vs {r["reference"]} | {[round(x*100,4) for x in r["seed_deltas"]]} | {100*r["mean_delta"]:+.4f} | {[round(x*100,4) for x in r["patient95"]]} | {[round(x*100,4) for x in r["family_interval"]]} | {r["pair_supported"]} |')
lines+=['',f'问题一CNN更新食谱支持：{gates["cnn_update_supported"]}；问题二完整DINO系统支持：{gates["dino_system_supported"]}。后者三个比较必须全部支持，不用单项替代。',
    'CNN低LR与clip同时改变，是整体食谱而非LR单一因果。DINO与CNN预训练、架构、特征维数和计算不同，约12.25G对7.33G计数FLOPs，非等预算或单一自监督贡献。',
    'DINO使用原官方权重经timm位置插值，非官方精确推理；公开数据重叠未审计。AC按CNN一致性构建可能偏向CNN；等级不是医生真值。',
    'val同时选best且重复开发，条件区间未覆盖所有选择、训练与预训练不确定性；不是独立临床确认。所有阴性保留，不重筛AC清单、不改标签；最终候选冻结后统一评价固定ACtest576。',
    '独立审计重建12模型全部427图推理，双权重/签名/SHA/60ep/最早最高val/LR/BN/冻结源/漂移/有限参数/预算/概率/所有权核验通过。',
    f'六任务窗口GPU均值{resource["mean_gpu_percent"]:.2f}%，CPU{resource["mean_cpu_percent"]:.2f}%，I/O等待{resource["mean_iowait_percent"]:.4f}%；不将分析准备阶段称为持续满载。']
(s/'ROUND3_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
assert all(p.owner()=='login' for p in s.rglob('*') if p.is_file())
subprocess.run([core.GIT,'-C',str(ROOT),'add','experiments/audit_ac_backbone_recipe.py',str(s.relative_to(ROOT))],check=True)
subprocess.run([core.GIT,'-C',str(ROOT),'diff','--cached','--check'],check=True)
subprocess.run([core.GIT,'-C',str(ROOT),'commit','-m','results: independently audit AC backbone and update-recipe benchmark'],check=True)
core.save_json(s/'status.json',dict(phase='independently_audited',updated_at=core.timestamp(),completed=9,**gates))
print(json.dumps(dict(passed=True,summary=summary,comparisons=comparisons,**gates,resource=resource)),flush=True)
