"""Strict actual weights; exact evaluate.run_inference timing helper on fixed val."""
from pathlib import Path
import ast, gc, hashlib, json, os, sys, time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
F=Path(__file__).resolve().parent;R=F.parents[1];S=R/'experiments/ac_focal_pairing_dropmom_control_window01'
sys.path.insert(0,str(S));import run_module_structure as m

def main():
    cfg=json.loads((F/'config.json').read_text());assert cfg['sources'][Path(__file__).name]==m.sha(__file__)
    assert m.READ(S/'independent_audit.json')['passed'] and m.READ(S/'bootstrap_complete.json')['passed']
    m.verify();torch.set_num_threads(1)
    reference=R/'experiments/ac_evaluate_metric_reaudit_window01/evaluate_reference.py';assert m.sha(reference)==cfg['evaluate_sha256']
    node=next(n for n in ast.parse(reference.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run_inference')
    namespace=dict(globals());exec(compile(ast.Module(body=[node],type_ignores=[]),str(reference),'exec'),namespace);run=namespace['run_inference']
    indices=m.READ(S/'config.json')['indices']['val'];records={v['index']:v for v in m.READ(R/'experiments/ac_dataset_v1/dataset_manifest.json')}
    paths=[R/records[i]['ac_path'] for i in indices]
    assert len(paths)==427 and all(m.sha(p)==records[i]['sha256'] for i,p in zip(indices,paths))
    transform=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    units=[('B0','plain',m.READ(S/'baseline_reference.json')['units'][0]),('Mom','mom_only',m.READ(S/'working_reference.json')['units'][0]),('Selected','mom_misaligned',m.READ(S/'selected_reference.json')['units'][0]),('DropMom','misaligned_only',m.READ(S/'runs/fdm_misaligned_only_s45/complete.json'))]
    results=[]
    for label,arm,u in units:
        assert m.sha(R/u['checkpoint'])==u['checkpoint_sha256'];model=m.model_for(45,arm).eval();state=torch.load(R/u['checkpoint'],map_location='cpu',weights_only=True)
        (model.backbone if arm=='plain' else model).load_state_dict(state,strict=True);del state
        # Same physical B32 FP32 for all four; no training/profile context remains.
        x=torch.stack([transform(Image.open(p).convert('RGB')) for p in paths[:32]]).cuda()
        with torch.inference_mode():
            for _ in range(10):model(x)
        torch.cuda.synchronize();del x
        measures=[]
        for _ in range(3):
            before=m.engine.snapshot();probs,eff=run(model,paths,transform,32,torch.device('cuda'));after=m.engine.snapshot()
            assert probs.shape==(427,3) and np.isfinite(probs).all();measures.append(dict(efficiency=eff,resources_before=before,resources_after=after))
        budget=next(b for b in m.READ(S/'preflight.json')['budgets'] if b['arm']==arm)
        assert sum(p.numel() for p in model.parameters())==budget['parameters']
        results.append(dict(name=label,arm=arm,seed=45,checkpoint=u['checkpoint'],checkpoint_sha256=u['checkpoint_sha256'],parameters=budget['parameters'],MACs=budget['counted_flops']/2,FLOPs=budget['counted_flops'],pure_fps_median=float(np.median([z['efficiency']['pure_fps'] for z in measures])),e2e_fps_median=float(np.median([z['efficiency']['e2e_fps'] for z in measures])),runs=measures))
        del model;gc.collect();torch.cuda.empty_cache()
    m.SAVE(F/'efficiency_metrics.json',dict(passed=True,actual_weights_strictloaded=True,evaluate_run_inference_exact_AST=True,evaluate_sha256=cfg['evaluate_sha256'],timing_source_sha256=m.sha(__file__),val_indices_sha256=m.hashlib.sha256(json.dumps(indices).encode()).hexdigest(),cases=427,split='val',test_images=0,new_training=0,physical_batch=32,precision='FP32; screenshot evaluate helper has no AMP; distinct from original AMP classification probabilities',transform='original research squareResize256 center224; not replacing frozen classification predictions',repetitions=3,warmup_batches=10,seed=45,results=results,flops_convention='2 FLOP/MAC all module conv/linear plus3 functional Mom rotated conv; excludes LN/GELU/pointwise/padding/memory; not uncorrected thop',scope='host-load-dependent computational efficiency; never classification+1.5pp'))

if __name__=='__main__':main()
