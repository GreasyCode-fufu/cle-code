def preflight():
    import numpy as np,torch
    from torch.nn import functional as functional
    cfg=verify();bind();assert READ(S/'module_source_audit.json')['passed']
    assert sha(Q/'cache448/images.npy')==READ(Q/'cache448_meta.json')['sha256']=='630081da3e38487987c02b60ba78f306192cb0abb036fd4e2411e6a05279b072'
    assert sha(T/'cache/test448.npy')==READ(T/'cache_audit.json')['sha256']
    assert sha(R/'.cache_data_study/resized_uint8.npy')==READ(R/'.cache_data_study/meta.json')['cache_sha256']
    x,y,_,idx=next(iter(loader(cfg['indices']['train'],True,45)));x,y=x.cuda(),y.cuda();assert x.shape==(32,3,384,384);plain=model_for(45,'plain').eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):expected=plain(x).clone()
    del plain;torch.cuda.empty_cache();budgets=[];diagnostics=[]
    for arm in ['plain',*ARMS]:
        model=model_for(45,arm).eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):pred=model(x)
        assert torch.equal(pred,expected),arm
        opt=optimizer(model);ids=[id(p) for g in opt.param_groups for p in g['params']];assert len(ids)==len(set(ids))==len(list(model.parameters()));flops=counted_flops(model,x[:1],base_flops);model.train();torch.cuda.reset_peak_memory_stats();scaler=torch.amp.GradScaler('cuda',init_scale=128.);gradients=[]
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=functional.cross_entropy(model(x),y)
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),arm
            assert any(p.grad is not None and float(p.grad.norm())>0 for p in opt.param_groups[0]['params'])
            projection={}
            if arm not in ['plain','local_only']:
                projection['moment']=float(model.encoder[0][0][0].adapter.weight.grad.norm());assert projection['moment']>0
            if arm not in ['plain','mom_only']:
                projection['merge']=float(model.encoder[0][2].adapter.weight.grad.norm());assert projection['merge']>0
            gradients.append(dict(step=step,CE_loss=float(loss.detach()),new_projection_gradient_norms=projection))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);scaler.step(opt);scaler.update()
        diagnostics.append(dict(arm=arm,gradients=gradients,actual_train_images=len(idx),test_images=0))
        budgets.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),counted_flops=flops,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,finite_two_updates=True));del model,opt,pred,scaler,loss;gc.collect();torch.cuda.empty_cache()
    del x,y,expected;reproduced=[]
    for kind,units in [('parent_Swin_B',READ(S/'parent_reference.json')['units']),('fixed_B0_A',READ(S/'baseline_reference.json')['units'])]:
        for unit in units:
            assert sha(R/unit['checkpoint'])==unit['checkpoint_sha256']
            model=model_for(unit['seed'],'plain') if kind=='parent_Swin_B' else q.model_for(unit['seed'],'convnext_tiny_A')
            state=torch.load(R/unit['checkpoint'],map_location='cpu',weights_only=True);(model.backbone if kind=='parent_Swin_B' else model).load_state_dict(state,strict=True);del state
            dl=loader(cfg['indices']['val'],False,unit['seed']) if kind=='parent_Swin_B' else torch.utils.data.DataLoader(q.GuardedData(cfg['indices']['val'],False,'A'),batch_size=32,shuffle=False,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(unit['seed']))
            p,yy,ii=core.infer(model,dl);saved=list(csv.DictReader(((R/unit['complete']).parent/'validation_predictions.csv').open()));target=np.array([[float(z[f'p{k}']) for k in range(3)] for z in saved]);error=float(np.max(np.abs(p-target)));assert ii==cfg['indices']['val'] and error<1e-6;reproduced.append(dict(kind=kind,seed=unit['seed'],full427_max_probability_error=error));del model,dl;gc.collect();torch.cuda.empty_cache()
    try:loader(cfg['indices']['test'][:1],False,45)
    except AssertionError:pass
    else:raise AssertionError('training loader accepted test')
    SAVE(S/'preflight.json',dict(passed=True,budgets=budgets,descriptor_diagnostics=diagnostics,reproduced_baselines=reproduced,test_denied=True,objective='original CE only',B_cache_original_images_not256_upsampling=True,cache_B_train2077_val427_test0=True,flops_convention='2FLOP/MAC includes functional Swin attention,3 rotated stem convolutions and new merge linear; excludes LN/statistics/pointwise/memory, actual throughput separately measured'))
    commit('test: verify Swin384 fullbatch CUDA pairing two updates moment merge gradients and all6 parent B0 full427 reconstructions')
