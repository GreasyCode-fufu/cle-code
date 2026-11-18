def preflight():
    import numpy as np,torch
    from torch.nn import functional as functional
    cfg=verify();bind();assert READ(S/'module_source_audit.json')['passed']
    assert sha(R/'.cache_data_study/resized_uint8.npy')==READ(R/'.cache_data_study/meta.json')['cache_sha256']
    x,y,_,idx=next(iter(loader(cfg['indices']['train'],True,45)));x,y=x.cuda(),y.cuda();budgets=[];diagnostics=[]
    reference=model_for(45,'plain').eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):expected=reference(x).clone()
    del reference;torch.cuda.empty_cache()
    vx,_,_,vidx=next(iter(loader(cfg['indices']['val'],False,45)));vx=vx.cuda()
    for arm in ['plain',*ARMS]:
        model=model_for(45,arm).eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):pred=model(x)
        assert torch.equal(pred,expected),arm
        opt=optimizer(model);ids=[id(p) for g in opt.param_groups for p in g['params']];assert len(ids)==len(set(ids))==len(list(model.parameters()))
        flops=counted_flops(model,x[:1],base_flops);stats=[]
        if model.readout is not None:
            def record(module,args):
                with torch.no_grad():
                    d=module.describe(*args);stats.append(dict(descriptor_l2=float(d.norm(dim=1).mean()),descriptor_rms=float(d.square().mean().sqrt()),descriptor_max_abs=float(d.abs().max())))
            hook=model.readout.register_forward_pre_hook(record)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):model(x);model(vx)
            hook.remove();assert all(0<r['descriptor_rms']<=1.01 for r in stats)
        model.train();torch.cuda.reset_peak_memory_stats();scaler=torch.amp.GradScaler('cuda',init_scale=128.);grads=[]
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=functional.cross_entropy(model(x),y)
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),arm
            assert any(p.grad is not None and float(p.grad.abs().sum())>0 for p in opt.param_groups[0]['params'])
            if model.readout is not None:
                g={name:float(p.grad.norm()) if p.grad is not None else 0. for name,p in model.readout.named_parameters()}
                assert g['classifier.weight']>0
                if step==1:assert g['low.weight']>0 and g['high.weight']>0,arm
                grads.append(dict(step=step,loss=float(loss),gradient_norms=g))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);scaler.step(opt);scaler.update()
        if model.readout is not None:
            model.eval();parts=[]
            hook=model.readout.register_forward_hook(lambda module,args,out:parts.append(out.detach().float()))
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):model(vx)
            hook.remove();assert torch.isfinite(parts[0]).all() and float(parts[0].abs().max())>0
            diagnostics.append(dict(arm=arm,train_images=len(idx),val_images=len(vidx),test_images=0,initial_train_descriptor=stats[0],initial_val_descriptor=stats[1],two_update_gradient_norms=grads,post_two_update_val_residual_rms=float(parts[0].square().mean().sqrt()),scale=16,scale_policy='fixed sqrt256 dimension, not selected using metric scores'))
        budgets.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),counted_flops=flops,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,finite_two_updates=True))
        del model,opt,pred,loss,scaler;gc.collect();torch.cuda.empty_cache()
    del x,y,vx,expected
    reproduced=[]
    for kind,units in [('plain',READ(S/'baseline_reference.json')['units']),('moments',[u for u in READ(S/'working_reference.json')['units'] if u['arm']=='c4_moments'])]:
        for unit in units:
            assert sha(R/unit['checkpoint'])==unit['checkpoint_sha256'];model=model_for(unit['seed'],kind);state=torch.load(R/unit['checkpoint'],map_location='cpu',weights_only=True)
            (model.backbone if kind=='plain' else model).load_state_dict(state,strict=True);del state
            p,yy,ii=core.infer(model,loader(cfg['indices']['val'],False,unit['seed']));saved=list(csv.DictReader(((R/unit['complete']).parent/'validation_predictions.csv').open()))
            target=np.array([[float(z[f'p{k}']) for k in range(3)] for z in saved]);error=float(np.max(np.abs(p-target)));assert ii==cfg['indices']['val'] and error<1e-6
            reproduced.append(dict(kind=kind,seed=unit['seed'],full427_max_probability_error=error));del model;gc.collect();torch.cuda.empty_cache()
    try:loader(cfg['indices']['test'][:1],False,45)
    except AssertionError:pass
    else:raise AssertionError('training loader accepted test')
    SAVE(S/'preflight.json',dict(passed=True,budgets=budgets,readout_scale_and_gradient_diagnostics=diagnostics,reproduced_baselines=reproduced,test_denied=True,flops_convention='2FLOP/MAC, Conv/Linear/functional3C4/convolution+bmm; excludes pointwise/norm/pooling'))
    commit('test: validate cross-scale CUDA AMP gradients, nondegenerate train-val scales and six full frozen reference probability reconstructions')
