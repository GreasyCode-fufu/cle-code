"""Independent round10 completion audit; inference only, never trains."""
import csv
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_token_adapter_study as study
from pretrained_representations import dino
from models.optimize.cle_token_adapter import LocalTokenAdapter
core=study.core;cfg=study.verify();s=study.STUDY
records=core.read_json(s/'manifest.json');jobs=core.read_json(s/'final_jobs.json')
assert len(jobs)==18
assert core.digest(core.CACHE/'resized_uint8.npy')==core.read_json(core.CACHE/'meta.json')['cache_sha256']

def rows(path):
    with open(path) as f:return list(csv.DictReader(f))

canonical=dino(ROOT)
initial={k:v.detach().clone() for k,v in canonical.state_dict().items()}
del canonical
predictions={};audit=[];curves=[];group_names=sorted({records[i]['group_id'] for i in cfg['val_indices']})
for job in jobs:
    assert core.complete(job)
    out=s/'runs'/job['id'];comp=core.read_json(out/'complete.json');result=core.read_json(out/'validation_complete.json')
    assert comp['signature']==result['signature']==core.signature(job)
    assert comp['checkpoint_sha256']==result['checkpoint_sha256']==core.digest(ROOT/comp['checkpoint'])
    log=rows(out/'train_log.csv')
    assert [int(row['epoch']) for row in log]==list(range(60)) and comp['epochs_completed']==60
    assert all(np.isfinite(float(v)) for row in log for v in row.values())
    best=int(np.argmax([float(row['val_macro_f1']) for row in log]))
    assert best==comp['best_epoch'] and abs(float(log[best]['val_macro_f1'])-comp['best_select_f1'])<1e-12
    assert set(job['train'])==set(range(2386)) and not job['sample_weights']
    assert set(job['select'])==set(job['score'])==set(cfg['val_indices'])
    assert {records[i]['split'] for i in job['score']}=={'val'}
    assert not {records[i]['group_id'] for i in job['train']} & {records[i]['group_id'] for i in job['score']}
    rr=rows(out/'validation_predictions.csv');ids=[int(row['index']) for row in rr]
    assert ids==job['score'] and len(set(ids))==538
    saved=np.array([[float(row[f'p{k}']) for k in range(3)] for row in rr])
    labels=np.array([int(row['label']) for row in rr])
    assert np.isfinite(saved).all() and (saved>=0).all() and np.allclose(saved.sum(1),1,atol=2e-6)
    assert np.array_equal(saved.argmax(1),[int(row['pred']) for row in rr])
    for row in rr:
        record=records[int(row['index'])]
        assert row['group_id']==record['group_id'] and row['sample_id']==record['sample_id'] and int(row['label'])==record['label']
    metric=core.metrics(saved,labels)
    assert abs(metric['macro_f1']-result['macro_f1'])<1e-12 and metric['confusion_matrix']==result['confusion_matrix']
    assert abs(metric['macro_f1']-comp['best_select_f1'])<1e-12
    model=study.model_for(job['seed'],job['variant'])
    state=torch.load(ROOT/comp['checkpoint'],map_location='cpu',weights_only=True)
    assert all(torch.isfinite(v).all() for v in state.values())
    model.load_state_dict(state,strict=True)
    source=model.source_state();squared=count=0
    for name,tensor in source.items():
        if job['variant']!='full_finetune':assert torch.equal(tensor.cpu(),initial[name]),name
        squared+=float((tensor.detach().cpu().double()-initial[name].double()).square().sum());count+=tensor.numel()
    rms=(squared/count)**.5
    assert (rms>0)==(job['variant']=='full_finetune')
    trainable=sum(v.numel() for v in model.parameters() if v.requires_grad)
    assert trainable==comp['trainable_parameters']==result['trainable_parameters']
    descriptors=[];contexts=[]
    head=model.head.register_forward_hook(lambda module,args,output:descriptors.append(args[0].detach().float().cpu().numpy()))
    def capture_context(module,args,output):
        a,b=output.detach().float().chunk(2,-1)
        contexts.append(torch.stack((1+.5*a.tanh().mean(-1),.5*b.tanh().mean(-1)),-1).cpu().numpy())
    handles=[m.context.register_forward_hook(capture_context) for m in model.modifiers if isinstance(m,LocalTokenAdapter) and m.context is not None]
    p,y,idx=core.infer(model,core.loader(job['score'],False,seed=job['seed']))
    maxdiff=float(np.max(np.abs(saved-p)))
    assert maxdiff<2e-6 and np.array_equal(saved.argmax(1),p.argmax(1)) and np.array_equal(y,labels) and idx==ids
    on=np.concatenate(descriptors);descriptors.clear()
    for h in handles:h.remove()
    model.enable_modifiers(False)
    po,yo,io=core.infer(model,core.loader(job['score'],False,seed=job['seed']))
    off=np.concatenate(descriptors);head.remove()
    assert io==ids and np.array_equal(yo,labels)
    relative=np.linalg.norm(on-off,axis=1)/(np.linalg.norm(off,axis=1)+1e-8)
    diag=result['diagnostics']
    assert abs(float(relative.mean())-diag['descriptor_relative_change_mean'])<2e-6
    assert abs(float(np.mean(p.argmax(1)!=po.argmax(1)))-diag['off_prediction_disagreement'])<1e-12
    assert core.metrics(po,yo)['confusion_matrix']==diag['off_same_checkpoint_metrics']['confusion_matrix']
    if contexts:
        cr=rows(out/'context_diagnostics.csv');assert [int(row['index']) for row in cr]==ids
        pos=0
        for start in range(0,538,32):
            for block,values in zip([8,9,10,11],contexts[pos:pos+4]):
                assert np.isfinite(values).all() and (values[:,0]>.5).all() and (values[:,0]<1.5).all()
                assert (np.abs(values[:,1])<.5).all()
                for k,value in enumerate(values):
                    assert abs(float(cr[start+k][f'block{block}_mean_scale'])-float(value[0]))<2e-6
                    assert abs(float(cr[start+k][f'block{block}_mean_bias'])-float(value[1]))<2e-6
            pos+=4
        assert pos==len(contexts)
    matrix=np.zeros((39,3,3))
    np.add.at(matrix,([group_names.index(row['group_id']) for row in rr],labels,saved.argmax(1)),1)
    predictions[job['variant'],job['seed']]=matrix
    audit.append(dict(job=job['id'],checkpoint_sha256=comp['checkpoint_sha256'],source_rms_drift=rms,
                      trainable_parameters=trainable,prediction_max_absolute_diff=maxdiff,
                      descriptor_relative_change=float(relative.mean()),adapter_off_disagreement=diag['off_prediction_disagreement']))
    curves.append(dict(variant=job['variant'],seed=job['seed'],best_epoch=best,best_f1=metric['macro_f1'],
                       final_f1=float(log[-1]['val_macro_f1']),final_train_accuracy=float(log[-1]['train_acc']),
                       skipped_amp_steps=sum(int(row['amp_skipped_steps']) for row in log)))
    assert (ROOT/comp['checkpoint']).owner()=='login'
    assert all(f.owner()=='login' for f in out.rglob('*') if f.is_file())
    print('audited',job['id'],flush=True)
    del state,model,source;torch.cuda.empty_cache()

def f1(m):
    den=m.sum(-1)+m.sum(-2)
    return np.mean(np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),den,out=np.zeros_like(den),where=den!=0),axis=-1)

weights=np.random.default_rng(20261024).multinomial(39,np.ones(39)/39,size=20000)
comparisons=core.read_json(s/'paired_comparisons.json');gates=[]
for control,threshold in cfg['gate_minimum_deltas'].items():
    ds=[];boot=[]
    for seed in cfg['seeds']:
        a,b=predictions[control,seed],predictions[cfg['primary'],seed]
        ds.append(float(f1(b.sum(0))-f1(a.sum(0))))
        boot.append(f1(np.einsum('bg,gij->bij',weights,b))-f1(np.einsum('bg,gij->bij',weights,a)))
    distribution=np.mean(boot,axis=0);family=np.percentile(distribution,[.5,99.5])
    r=comparisons['comparisons'][cfg['primary']+'_vs_'+control]
    assert np.allclose(ds,r['seed_deltas'],atol=1e-12)
    assert np.allclose(np.percentile(distribution,[2.5,97.5]),r['patient95'],atol=1e-12)
    assert np.allclose(family,r['family_interval'],atol=1e-12)
    gate=np.mean(ds)>0 and np.mean(ds)>=threshold and family[0]>0 and (control!='frozen_head' or min(ds)>0)
    assert bool(gate)==r['strong_pair'];gates.append(gate)
assert bool(all(gates))==comparisons['strong_primary_followup']
summary=core.read_json(s/'round10_summary.json')
for r in summary:
    matrices=[predictions[r['variant'],seed] for seed in cfg['seeds']]
    assert abs(np.mean([f1(m.sum(0)) for m in matrices])-r['mean_validation_f1'])<1e-12
    assert abs(np.mean([m.sum(0)[1,1]/m.sum(0)[1].sum() for m in matrices])-r['mean_malignant_recall'])<1e-12
    assert abs(np.mean([np.mean(np.trace(m,axis1=-2,axis2=-1)/m.sum((1,2))) for m in matrices])-r['mean_patient_accuracy'])<1e-12
core.save_json(s/'completion_audit.json',dict(passed=True,timestamp=core.timestamp(),new_neural_trainings=18,
    checks=['all source/cache/model hashes, run signatures and 60 epochs','inner best checkpoint agrees with log and full-val metrics',
            'full train2386/val538 patient isolation, no test or grading','18 strict checkpoint reloads and full-val on/off inference reproduce predictions',
            'pretrained source tensors remain exact except full fine-tuning, source drift recorded','all weights finite, trainable budgets match',
            'per-block per-image context finite, bounded and reproduced','39-patient bootstrap, five family intervals and complete gate',
            'all six scheme summaries and login ownership'],models=audit))
core.save_json(s/'training_curve_audit.json',curves)
print('PASS independent round10 audit',flush=True)
