def preflight():
    import numpy as np,torch
    from torch.nn import functional as functional
    cfg=verify();bind();assert READ(S/'module_source_audit.json')['passed'];assert sha(R/'.cache_data_study/resized_uint8.npy')==READ(R/'.cache_data_study/meta.json')['cache_sha256']
    cache_audit=READ(T/'cache_audit.json');assert cache_audit['passed'] and sha(T/'cache/test448.npy')==cache_audit['sha256']
    x,y,_,idx=next(iter(loader(cfg['indices']['train'],True,45)));x,y=x.cuda(),y.cuda();plain=model_for(45,'plain').eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):expected=plain(x).clone()
    del plain;torch.cuda.empty_cache();budgets=[];diagnostics=[]
    for arm in ['plain','mom_only',*ARMS]:
        model=model_for(45,arm).eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):pred=model(x)
        assert torch.equal(pred,expected),arm
        opt=optimizer(model);ids=[id(p) for g in opt.param_groups for p in g['params']];assert len(ids)==len(set(ids))==len(list(model.parameters()));assert model.midstage is None or {id(p) for p in model.midstage.parameters()}<={id(p) for p in opt.param_groups[1]['params']};assert opt.param_groups[0]['lr']==1e-5 and opt.param_groups[1]['lr']==1e-4;flops=counted_flops(model,x[:1],base_flops);model.train();torch.cuda.reset_peak_memory_stats();scaler=torch.amp.GradScaler('cuda',init_scale=128.);gradients=[]
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=functional.cross_entropy(model(x),y)
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),arm
            assert any(p.grad is not None and float(p.grad.norm())>0 for p in opt.param_groups[0]['params'])
            if arm in ARMS:
                up_norm=float(model.midstage.up.weight.grad.norm());upstream={n:float(p.grad.norm()) for n,p in model.midstage.named_parameters() if n!='up.weight'}
                assert up_norm>0 and all(v==0. if step==0 else v>0 for v in upstream.values()),(arm,step,up_norm,upstream)
                moment_norm=float(model.encoder[0][0][0].adapter.weight.grad.norm()) if arm!='focalcontrast_only' else None
                if moment_norm is not None:assert moment_norm>0
                gradients.append(dict(step=step,CE_loss=float(loss.detach()),zero_up_gradient_norm=up_norm,upstream_gradient_norms=upstream,retained_moment_gradient_norm=moment_norm,expected_upstream_first_zero_second_positive=True))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);scaler.step(opt);scaler.update()
        if arm in ARMS:diagnostics.append(dict(arm=arm,gradients=gradients,actual_train_images=len(idx),test_images=0))
        budgets.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),counted_flops=flops,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,finite_two_updates=True));del model,opt,pred,scaler,loss;gc.collect();torch.cuda.empty_cache()
    assert len(idx)==32 and len(set(idx.tolist()))==32 and set(idx.tolist())<=set(cfg['indices']['train'])
    del x,y,expected;reproduced=[]
    for kind,units in [('plain',READ(S/'baseline_reference.json')['units']),('mom_only',[u for u in READ(S/'working_reference.json')['units'] if u['arm']=='c4_moments'])]:
        for unit in units:
            assert sha(R/unit['checkpoint'])==unit['checkpoint_sha256'];model=model_for(unit['seed'],kind);state=torch.load(R/unit['checkpoint'],map_location='cpu',weights_only=True);(model.backbone if kind=='plain' else model).load_state_dict(state,strict=True);del state
            p,yy,ii=core.infer(model,loader(cfg['indices']['val'],False,unit['seed']));saved=list(csv.DictReader(((R/unit['complete']).parent/'validation_predictions.csv').open()));target=np.array([[float(z[f'p{k}']) for k in range(3)] for z in saved]);error=float(np.max(np.abs(p-target)));assert ii==cfg['indices']['val'] and error==0.;reproduced.append(dict(kind=kind,seed=unit['seed'],full427_max_probability_error=error));del model;gc.collect();torch.cuda.empty_cache()
    try:loader(cfg['indices']['test'][:1],False,45)
    except AssertionError:pass
    else:raise AssertionError('training loader accepted test')
    SAVE(S/'preflight.json',dict(passed=True,budgets=budgets,descriptor_diagnostics=diagnostics,reproduced_baselines=reproduced,test_denied=True,objective='original CE only; patient auxiliary targets not retained as confirmed increments',flops_convention='2FLOP/MAC includes3 functional rotated convolutions,all new1x1 and depthwiseConv; excludes LN/GELU/centering/elementwise-gating/replicate-padding/memory, actual throughput separately measured'))
    commit('test: verify midstage focal contrast controls CUDA AMP two-step chain-rule gradients pairing and all6 frozen full427 probability reconstructions')
