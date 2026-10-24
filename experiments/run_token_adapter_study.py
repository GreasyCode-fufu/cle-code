#!/usr/bin/env python3
"""Round10: controlled pretrained-token CLE adaptation, development only."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments')]
import run_architecture_study as arch
from pretrained_representations import DINO_SHA, runtime, state_sha
from models.optimize.cle_token_adapter import CLETokenAdapterNet, VARIANTS, LoRAQKV, LocalTokenAdapter, counted_flops

core = arch.core
DATA = ROOT/'experiments/data_study'
STUDY = ROOT/'experiments/token_adapter_study'
MODEL = ROOT/'models/optimize/cle_token_adapter.py'
arch.STUDY = STUDY
arch.__file__ = __file__
core.STUDY = STUDY
core.WEIGHTS = ROOT/'weights/token_adapter_study'
SAVE = core.save_json
COMMIT = arch.commit


def save_json(path, value):
    if Path(path) == STUDY/'status.json' and value.get('phase') == 'round1_training':
        value = dict(value, phase='round10_token_adaptation', round=10, new_neural_trainings=18)
    SAVE(path, value)


core.save_json = save_json
arch.commit = lambda message: COMMIT(message.replace('development-only architecture run', 'CLE token-adaptation run'))


def sources():
    from timm.layers.attention import Attention
    return dict(driver=core.digest(__file__), model=core.digest(MODEL),
                loader=core.digest(ROOT/'experiments/pretrained_representations.py'),
                protocol=core.digest(ROOT/'experiments/ROUND10_PROTOCOL.md'),
                trainer=core.digest(core.__file__), supervisor=core.digest(ROOT/'experiments/run_architecture_study.py'),
                backbone=core.digest(ROOT/'cle_fair_common.py'), runtime=runtime(),
                attention=core.digest(inspect.getsourcefile(Attention)))


def verify():
    cfg = core.read_json(STUDY/'config.json')
    assert cfg['architecture_sources'] == sources()
    for path, key in [(DATA/'config.json', 'data_config_sha256'), (STUDY/'manifest.json', 'manifest_sha256'),
                      (STUDY/'final_jobs.json', 'jobs_sha256'), (core.CACHE/'meta.json', 'cache_meta_sha256')]:
        assert core.digest(path) == cfg[key]
    assert core.digest(ROOT/'weights/pretrained/dinov2_vits14_pretrain.pth') == DINO_SHA == cfg['pretrained_sha256']
    return cfg


arch.verify = verify
arch.baseline_predictions = lambda: None


def prepare():
    STUDY.mkdir(parents=True, exist_ok=True)
    if (STUDY/'config.json').exists():
        verify()
        return
    old = core.read_json(DATA/'config.json')
    (STUDY/'manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
    records = core.read_json(STUDY/'manifest.json')
    train = [i for i, row in enumerate(records) if row['split'] == 'train']
    val = old['val_indices']
    assert len(train) == 2386 and len(val) == 538
    assert not {records[i]['group_id'] for i in train} & {records[i]['group_id'] for i in val}
    jobs = []
    for variant in VARIANTS:
        for seed in [42, 43, 44]:
            jobs.append(dict(id=f'r10_{variant}_s{seed}', phase='token_adaptation', arm=variant, variant=variant,
                             seed=seed, train=train, select=val, score=val, epochs=60, recipe='full_unweighted',
                             sample_weights={}, architecture_sources=sources()))
    core.save_json(STUDY/'final_jobs.json', jobs)
    cfg = dict(round=10, created_at=core.timestamp(), architecture_sources=sources(), variants=VARIANTS, seeds=[42,43,44],
               primary='cle_cls_context', new_neural_trainings=18, pretrained_sha256=DINO_SHA,
               data_config_sha256=core.digest(DATA/'config.json'), manifest_sha256=core.digest(STUDY/'manifest.json'),
               jobs_sha256=core.digest(STUDY/'final_jobs.json'), cache_meta_sha256=core.digest(core.CACHE/'meta.json'),
               train_indices=train, val_indices=val, effective_indices=train+val, epochs=60, batch_size=32,
               encoder_lr_full=1e-5, head_adapter_lr=1e-4, weight_decay=.05, gradient_clip_norm=1., scheduler='cosine to0',
               optimizer='AdamW', amp='float16 with original dynamic GradScaler', recipe='full_unweighted',
               image_transform='Original resize256/crop224, H/V flip/rotate15, center eval, ImageNet normalization',
               extra_blocks=[8,9,10,11], lora_rank=24, lora_alpha=24, local_rank=48, context_rank=24,
               context_scale_bound=[.5,1.5], context_bias_bound=[-.5,.5], max_parallel=2, cpu_threads=2,
               bootstrap=dict(groups=39, draws=20000, seed=20261024, family_comparisons=5, percentiles=[.5,99.5]),
               gate_minimum_deltas=dict(frozen_head=.01,full_finetune=0.,lora_qv=.005,local_adapter=.005,cle_mean_context=.003),
               policy='Development only, full val, no test/grades/relabeling; exploratory after failed R9 full primary gate.')
    core.save_json(STUDY/'config.json', cfg)
    core.save_json(STUDY/'status.json', dict(phase='prepared',round=10,total=18,updated_at=core.timestamp()))
    arch.commit('research: freeze controlled CLE token-adaptation recipe and 18 paired training jobs')


def model_for(seed, variant):
    core.set_seed(seed)
    return CLETokenAdapterNet(ROOT, variant).cuda()


def optimizer_for(model):
    import torch
    if model.variant == 'full_finetune':
        groups = [dict(params=list(model.encoder.parameters()),lr=1e-5), dict(params=list(model.head.parameters()),lr=1e-4)]
    else:
        groups = [dict(params=[p for p in model.parameters() if p.requires_grad],lr=1e-4)]
    return torch.optim.AdamW(groups, weight_decay=.05)


def train_job(job):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score
    verify()
    model = model_for(job['seed'], job['variant'])
    out = STUDY/'runs'/job['id'];out.mkdir(parents=True, exist_ok=True)
    ckdir = core.WEIGHTS/job['id'];ckdir.mkdir(parents=True, exist_ok=True)
    ck = ckdir/'model_best.pth'
    train = core.loader(job['train'], True, seed=job['seed'])
    val = core.loader(job['select'], False, seed=job['seed'])
    opt = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=job['epochs'])
    scaler = torch.amp.GradScaler('cuda')
    best = -1.;best_epoch = -1;history = [];started = time.time()
    meta = dict(job_id=job['id'], signature=core.signature(job), seed=job['seed'], variant=job['variant'],
                train_count=len(job['train']), select_count=len(job['select']), created_at=core.timestamp(),
                git_commit=subprocess.check_output([core.GIT,'-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                pretrained_sha256=DINO_SHA, parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    core.save_json(out/'run_meta.json', meta)
    params = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(job['epochs']):
        model.train();total_loss=correct=n=skipped=0
        for x,y,_,_ in train:
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(x);loss=F.cross_entropy(logits,y)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite CE')
            previous_scale=scaler.get_scale()
            scaler.scale(loss).backward();scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params,1.)
            scaler.step(opt);scaler.update()
            skipped += int(scaler.get_scale()<previous_scale)
            total_loss+=float(loss.detach())*len(y);correct+=int((logits.argmax(1)==y).sum());n+=len(y)
        p,y,_=core.infer(model,val)
        val_f1=float(f1_score(y,p.argmax(1),labels=[0,1,2],average='macro',zero_division=0))
        scheduler.step()
        if val_f1>best:
            best,best_epoch=val_f1,epoch
            torch.save(model.state_dict(),ck.with_suffix('.tmp'));os.replace(ck.with_suffix('.tmp'),ck)
        history.append(dict(epoch=epoch,train_loss=total_loss/n,train_acc=correct/n,val_macro_f1=val_f1,
                            lr_first=opt.param_groups[0]['lr'],lr_head=opt.param_groups[-1]['lr'],
                            amp_skipped_steps=skipped,elapsed_seconds=round(time.time()-started,2)))
        core.save_csv(out/'train_log.csv',history)
        print(f'[{job["id"]}] epoch={epoch+1}/60 train_acc={correct/n:.4f} val_f1={val_f1:.4f} best={best:.4f}',flush=True)
    meta.update(completed_at=core.timestamp(),seconds=round(time.time()-started,2),best_select_f1=best,best_epoch=best_epoch,
                checkpoint=str(ck.relative_to(ROOT)),checkpoint_sha256=core.digest(ck),epochs_completed=len(history))
    core.save_json(out/'complete.json',meta)


def worker(job_id):
    import torch
    verify();job=next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id']==job_id)
    if not core.complete(job):train_job(job)
    out=STUDY/'runs'/job_id;comp=core.read_json(out/'complete.json')
    model=model_for(job['seed'],job['variant'])
    model.load_state_dict(torch.load(ROOT/comp['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    descriptors=[];context=[]
    def head_hook(module,args,output):descriptors.append(args[0].detach().float().cpu().numpy())
    def context_hook(module,args,output):
        a,b=output.detach().float().chunk(2,-1)
        context.append(torch.stack((1+.5*a.tanh().mean(-1),.5*b.tanh().mean(-1)),dim=-1).cpu().numpy())
    handle=model.head.register_forward_hook(head_hook)
    handles=[m.context.register_forward_hook(context_hook) for m in model.modifiers if isinstance(m,LocalTokenAdapter) and m.context is not None]
    p,y,idx=core.infer(model,core.loader(job['score'],False,seed=job['seed']))
    on=np.concatenate(descriptors);descriptors.clear()
    for h in handles:h.remove()
    core.save_csv(out/'validation_predictions.csv',core.prediction_rows(p,y,idx))
    assert abs(core.metrics(p,y)['macro_f1']-comp['best_select_f1'])<1e-10
    model.enable_modifiers(False)
    po,yo,io=core.infer(model,core.loader(job['score'],False,seed=job['seed']))
    off=np.concatenate(descriptors);handle.remove()
    assert idx==io and np.array_equal(y,yo)
    relative=np.linalg.norm(on-off,axis=1)/(np.linalg.norm(off,axis=1)+1e-8)
    # Context hooks were invoked in block order for each batch. Persist actual
    # per-block summaries, not localization or calibrated quality estimates.
    if context:
        rows=[];position=0
        for start in range(0,len(idx),32):
            size=min(32,len(idx)-start)
            block_values=context[position:position+4];position+=4
            for k in range(size):
                row=dict(index=idx[start+k])
                for block,values in zip([8,9,10,11],block_values):
                    row.update({f'block{block}_mean_scale':float(values[k,0]),f'block{block}_mean_bias':float(values[k,1])})
                rows.append(row)
        assert position==len(context)
        core.save_csv(out/'context_diagnostics.csv',rows)
    diag=dict(descriptor_relative_change_mean=float(relative.mean()),descriptor_relative_change_quantiles=np.percentile(relative,[0,25,50,75,100]).tolist(),
              off_prediction_disagreement=float(np.mean(p.argmax(1)!=po.argmax(1))),off_same_checkpoint_metrics=core.metrics(po,yo),
              probability_mean_absolute_change=float(np.abs(p-po).mean()),
              note='Same-checkpoint diagnostic, not retrained ablation, ROI truth or a noise diagnosis.')
    pf=core.read_json(STUDY/'preflight_validation.json');budget=next(r for r in pf['variants'] if r['variant']==job['variant'])
    core.save_json(out/'validation_complete.json',dict(signature=core.signature(job),job_id=job_id,variant=job['variant'],recipe=job['recipe'],
        seed=job['seed'],n=len(y),checkpoint_sha256=comp['checkpoint_sha256'],completed_at=core.timestamp(),
        parameters=comp['parameters'],trainable_parameters=comp['trainable_parameters'],budget=budget,diagnostics=diag,**core.metrics(p,y)))


def check():
    import torch
    import torch.nn.functional as F
    from pretrained_representations import dino
    cfg=verify()
    assert core.digest(core.CACHE/'resized_uint8.npy')==core.read_json(core.CACHE/'meta.json')['cache_sha256']
    records=core.read_json(STUDY/'manifest.json')
    assert all(records[i]['split'] in {'train','val'} for i in cfg['effective_indices'])
    assert len({records[i]['group_id'] for i in cfg['train_indices']})==185
    assert len({records[i]['group_id'] for i in cfg['val_indices']})==39
    # Independent algebra checks away from zero initialization.
    core.set_seed(42)
    low=LoRAQKV(torch.nn.Linear(8,24).double().requires_grad_(False),rank=3).double()
    with torch.no_grad():low.bq.normal_();low.bv.normal_()
    z=torch.randn(2,5,8,dtype=torch.float64,requires_grad=True)
    actual=low(z)
    reference=z@low.base.weight.T+low.base.bias
    reference=reference+torch.cat(((z@low.aq.T)@low.bq.T,torch.zeros_like(z),(z@low.av.T)@low.bv.T),-1)
    assert torch.allclose(actual,reference,atol=1e-10)
    variables=(z,low.aq,low.bq,low.av,low.bv)
    ga=torch.autograd.grad(actual.square().sum(),variables,retain_graph=True)
    gr=torch.autograd.grad(reference.square().sum(),variables)
    assert all(torch.allclose(a,b,atol=1e-8) for a,b in zip(ga,gr))
    local=LocalTokenAdapter(channels=8,rank=3,context='cls').double()
    with torch.no_grad():local.up.weight.normal_();local.context.weight.normal_(std=.1)
    t=torch.randn(2,10,8,dtype=torch.float64,requires_grad=True)
    u=F.layer_norm(t,(8,));v=F.gelu(F.linear(u[:,1:],local.down.weight,local.down.bias))
    images=v.transpose(1,2).reshape(2,3,3,3)
    patches=F.unfold(images,3,padding=1).reshape(2,3,9,9)
    dw=(patches*local.spatial.weight.reshape(1,3,9,1)).sum(2)+local.spatial.bias.reshape(1,3,1)
    a,b=F.linear(u[:,0],local.context.weight,local.context.bias).chunk(2,-1)
    dw=dw.transpose(1,2)*(1+.5*a.tanh().unsqueeze(1))+.5*b.tanh().unsqueeze(1)
    ref=torch.cat((torch.zeros_like(t[:,:1]),F.linear(F.gelu(dw),local.up.weight,local.up.bias)),1)
    fast=local(t);assert torch.allclose(fast,ref,atol=1e-10)
    assert torch.allclose(torch.autograd.grad(fast.square().sum(),t,retain_graph=True)[0],torch.autograd.grad(ref.square().sum(),t)[0],atol=1e-8)
    job=core.read_json(STUDY/'final_jobs.json')[0]
    x,y,_,_=next(iter(core.loader(job['train'][:32],False,seed=42)));x,y=x.cuda(),y.cuda()
    free=torch.cuda.mem_get_info()[0]
    canonical=model_for(42,'frozen_head').eval()
    initial={k:v.detach().cpu().clone() for k,v in canonical.encoder.state_dict().items()}
    head={k:v.detach().cpu().clone() for k,v in canonical.head.state_dict().items()}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):expected=canonical(x)
    del canonical;torch.cuda.empty_cache();checks=[]
    for kind in VARIANTS:
        model=model_for(42,kind).eval()
        assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in model.source_state().items())
        assert all(torch.equal(v.detach().cpu(),head[k]) for k,v in model.head.state_dict().items())
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):assert torch.equal(model(x),expected)
        flop=counted_flops(model,x[:1]);opt=optimizer_for(model)
        scaler=torch.amp.GradScaler('cuda',init_scale=128.)
        torch.cuda.reset_peak_memory_stats();model.train();gradient_steps=[]
        if kind!='full_finetune':assert not model.encoder.training
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(x),y)
            scaler.scale(loss).backward();scaler.unscale_(opt)
            names=[]
            for name,param in model.named_parameters():
                if not param.requires_grad:assert param.grad is None
                else:
                    assert param.grad is not None and torch.isfinite(param.grad).all(),name
                    if param.grad.abs().sum()>0:names.append(name)
            assert 'head.weight' in names
            if step==1 and kind=='lora_qv':
                assert all(any(name.endswith(suffix) for name in names) for suffix in ['aq','bq','av','bv'])
            if step==1 and kind in {'local_adapter','cle_cls_context','cle_mean_context'}:
                assert any('.down.weight' in name for name in names) and any('.spatial.weight' in name for name in names)
                if kind!='local_adapter':assert any('.context.weight' in name for name in names)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            scaler.step(opt);scaler.update();gradient_steps.append(len(names))
        peak=torch.cuda.max_memory_allocated()
        if kind!='full_finetune':
            assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in model.source_state().items())
        model.eval();model.enable_modifiers(False)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):off=model(x)
        if kind!='full_finetune':
            reference=dino(ROOT).cuda()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                tokens=reference.forward_features(x);correct=model.head(torch.cat((tokens[:,0],tokens[:,1:].mean(1)),dim=-1))
            assert torch.equal(off,correct)
            del reference
        checks.append(dict(variant=kind,parameters=sum(p.numel() for p in model.parameters()),
                           trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                           counted_flops_224=flop,amp_batch32_peak_bytes=peak,nonzero_gradient_counts_first_two_steps=gradient_steps))
        print('preflight',checks[-1],flush=True)
        del model,opt,scaler;torch.cuda.empty_cache()
    controls=[r for r in checks if r['variant'] in {'lora_qv','local_adapter','cle_cls_context','cle_mean_context'}]
    assert max(r['trainable_parameters'] for r in controls)/min(r['trainable_parameters'] for r in controls)<1.03
    assert max(r['counted_flops_224'] for r in controls)/min(r['counted_flops_224'] for r in controls)<1.01
    assert 2*max(r['amp_batch32_peak_bytes'] for r in checks)<.85*free
    core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),variants=checks,
        checks=['source/runtime/cache SHA and effective train/val indices','paired pretrained encoder and head, exact epoch0 AMP outputs',
                'LoRA forward/backward independent algebra','local reshape/depthwise convolution via independent unfold and input gradient',
                'two real-image AMP steps finite and lower/context gradients enabled after output projection learns',
                'frozen source tensors unchanged; modifier-off equals source encoder plus fitted head',
                'PEFT trainable parameter ratio <1.03 and counted total FLOP ratio <1.01','actual batch32 peak memory safe for two workers'],
        flops_note='Conv/Linear plus attention QK/AV, 2/MAC; excludes activation/normalization/softmax/data movement.'))
    arch.commit('test: validate paired token models, adapter gradients, frozen sources and fair budgets')


def summarize():
    cfg=verify();jobs=core.read_json(STUDY/'final_jobs.json');results=[];matrices={}
    records=core.read_json(STUDY/'manifest.json');groups=sorted({records[i]['group_id'] for i in cfg['val_indices']})
    weight=np.random.default_rng(20261024).multinomial(len(groups),np.ones(len(groups))/len(groups),size=20000)
    def f1(m):
        d=m.sum(-1)+m.sum(-2)
        return np.mean(np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),d,out=np.zeros_like(d),where=d!=0),axis=-1)
    for job in jobs:
        out=STUDY/'runs'/job['id'];r=core.read_json(out/'validation_complete.json');results.append(r)
        rows=sorted(arch.read_csv(out/'validation_predictions.csv'),key=lambda r:int(r['index']))
        assert [int(r['index']) for r in rows]==sorted(cfg['val_indices'])
        m=np.zeros((len(groups),3,3))
        np.add.at(m,([groups.index(r['group_id']) for r in rows],[int(r['label']) for r in rows],[int(r['pred']) for r in rows]),1)
        matrices[job['variant'],job['seed']]=m
    comparisons={}
    for control,threshold in cfg['gate_minimum_deltas'].items():
        ds=[];bootstrap=[]
        for seed in cfg['seeds']:
            a,b=matrices[control,seed],matrices[cfg['primary'],seed]
            ds.append(float(f1(b.sum(0))-f1(a.sum(0))))
            bootstrap.append(f1(np.einsum('bg,gij->bij',weight,b))-f1(np.einsum('bg,gij->bij',weight,a)))
        boot=np.mean(bootstrap,axis=0);family=np.percentile(boot,[.5,99.5]).tolist()
        strong=np.mean(ds)>0 and np.mean(ds)>=threshold and family[0]>0
        if control=='frozen_head':strong=strong and min(ds)>0
        comparisons[cfg['primary']+'_vs_'+control]=dict(seed_deltas=ds,mean_delta=float(np.mean(ds)),
            patient95=np.percentile(boot,[2.5,97.5]).tolist(),family_interval=family,strong_pair=bool(strong))
    summary=[]
    for variant in VARIANTS:
        runs=[r for r in results if r['variant']==variant];cms=[matrices[variant,seed] for seed in cfg['seeds']]
        summary.append(dict(variant=variant,mean_validation_f1=float(np.mean([r['macro_f1'] for r in runs])),
            seed_sd=float(np.std([r['macro_f1'] for r in runs],ddof=1)),
            mean_malignant_recall=float(np.mean([m.sum(0)[1,1]/m.sum(0)[1].sum() for m in cms])),
            mean_patient_accuracy=float(np.mean([np.mean(np.trace(m,axis1=-2,axis2=-1)/m.sum((1,2))) for m in cms])),
            parameters=runs[0]['parameters'],trainable_parameters=runs[0]['trainable_parameters']))
    strong=all(r['strong_pair'] for r in comparisons.values())
    core.save_json(STUDY/'round10_results.json',results);core.save_json(STUDY/'round10_summary.json',summary)
    core.save_json(STUDY/'paired_comparisons.json',dict(comparisons=comparisons,strong_primary_followup=strong))
    lines=['# Round10: CLE token adaptation','',
           '18 neural trainings, full original train2386/val538. No test, grades or relabeling.',
           'Exploratory checkpoint-selected development results; conditional intervals do not remove repeated selection bias.','',
           '| Arm | Mean val F1 | Malignant recall | Patient equal accuracy | Trainable |','|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["variant"]} | {r["mean_validation_f1"]:.6f} | {r["mean_malignant_recall"]:.6f} | {r["mean_patient_accuracy"]:.6f} | {r["trainable_parameters"]} |')
    lines+=['',f'Complete preregistered structure gate: {strong}.','Five paired comparisons and family intervals in paired_comparisons.json. Independent audit and patient confirmation required.']
    (STUDY/'ROUND10_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    arch.commit('results: compare CLE context-local adaptation against frozen, full, LoRA and local controls')
    core.save_json(STUDY/'status.json',dict(phase='round10_complete_needs_analysis',round=10,completed=18,total=18,updated_at=core.timestamp()))


arch.summarize=summarize
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','worker','run']);parser.add_argument('--job')
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='check':check()
    elif args.command=='worker':worker(args.job)
    else:arch.run()
