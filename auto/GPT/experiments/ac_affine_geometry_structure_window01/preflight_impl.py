def preflight():
    import numpy as np,torch
    from torch.nn import functional as functional
    from models.optimize.ac_affine_geometry_window01.affine_geometry import affine_offsets
    cfg=verify();bind();assert READ(S/'module_source_audit.json')['passed'];assert sha(R/'.cache_data_study/resized_uint8.npy')==READ(R/'.cache_data_study/meta.json')['cache_sha256']
    x,y,_,idx=next(iter(loader(cfg['indices']['train'],True,45)));x,y=x.cuda(),y.cuda();plain=model_for(45,'plain').eval()
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
            if arm!='plain':
                layer=model.encoder[0][5][-1].block[0];norm=float(layer.predictor.weight.grad.norm());assert norm>0;gradients.append(dict(step=step,loss=float(loss.detach()),predictor_weight_gradient_norm=norm))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);scaler.step(opt);scaler.update()
        if arm!='plain':
            stored=[];layer=model.encoder[0][5][-1].block[0];hook=layer.predictor.register_forward_hook(lambda module,args,out:stored.append(out.detach().float()));model.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):model(x)
            hook.remove();raw=stored[0];offset=layer.offsets(raw);assert torch.isfinite(offset).all() and float(offset.abs().max())>0
            record=dict(arm=arm,gradients=gradients,offset_rms=float(offset.square().mean().sqrt()),offset_max_abs=float(offset.abs().max()),actual_train_images=len(idx),test_images=0)
            if arm!='mom_dcn':
                _,a=affine_offsets(raw,7,return_jacobian=True);det=a[:,0,0]*a[:,1,1]-a[:,0,1]*a[:,1,0];assert float(det.min())>0;record['local_determinant_range']=[float(det.min()),float(det.max())]
            diagnostics.append(record)
        budgets.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),counted_flops=flops,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,finite_two_updates=True));del model,opt,pred,scaler,loss;gc.collect();torch.cuda.empty_cache()
    del x,y,expected;reproduced=[]
    for kind,units in [('plain',READ(S/'baseline_reference.json')['units']),('moments',[u for u in READ(S/'working_reference.json')['units'] if u['arm']=='c4_moments'])]:
        for unit in units:
            assert sha(R/unit['checkpoint'])==unit['checkpoint_sha256'];model=model_for(unit['seed'],kind);state=torch.load(R/unit['checkpoint'],map_location='cpu',weights_only=True);(model.backbone if kind=='plain' else model).load_state_dict(state,strict=True);del state
            p,yy,ii=core.infer(model,loader(cfg['indices']['val'],False,unit['seed']));saved=list(csv.DictReader(((R/unit['complete']).parent/'validation_predictions.csv').open()));target=np.array([[float(z[f'p{k}']) for k in range(3)] for z in saved]);error=float(np.max(np.abs(p-target)));assert ii==cfg['indices']['val'] and error<1e-6;reproduced.append(dict(kind=kind,seed=unit['seed'],full427_max_probability_error=error));del model;gc.collect();torch.cuda.empty_cache()
    try:loader(cfg['indices']['test'][:1],False,45)
    except AssertionError:pass
    else:raise AssertionError('training loader accepted test')
    SAVE(S/'preflight.json',dict(passed=True,budgets=budgets,geometry_diagnostics=diagnostics,reproduced_baselines=reproduced,test_denied=True,cuda_offset_backward_may_use_atomic_reductions=True,flops_convention='2FLOP/MAC including two functional DCN linear contractions, predictor and functional C4; excludes bilinear interpolation/pointwise/norm/pooling/memory, actual throughput must be measured'))
    commit('test: validate geometry CUDA AMP predictor gradients matched initialization and six full frozen427 probability reconstructions')
