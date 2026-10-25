"""Strict CLE checkpoint loading with the user's original evaluate.py metrics."""
from pathlib import Path
import argparse, ast, csv, hashlib, importlib.util, json, os, sys, time
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report

HERE=Path(__file__).resolve().parent
# Installed at R/experiments/ac_focal_pairing_dropmom_final_report_window01/delivery.
R=HERE.parents[2]
S=R/'experiments/ac_focal_pairing_dropmom_control_window01'
EA=R/'experiments/ac_evaluate_metric_reaudit_window01'
sys.path.insert(0,str(S));import run_module_structure as m
EVALUATE_SHA='e5213828f81e04baf2aa025c35dfa3a1b585efbdcf8a7cfbd9033510592b0ee7'

def clean(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:clean(z) for k,z in v.items()}
    if isinstance(v,(list,tuple)):return [clean(z) for z in v]
    return v

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['mom_misaligned','mom_only','plain','misaligned_only'],default='mom_misaligned')
    p.add_argument('--seed',type=int,choices=[45,46,47],default=45)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--split',choices=['val'],help='Fixed original427 validation cases; never silently scan another cohort')
    inputs.add_argument('--data_path',type=Path,help='Directory with the three original class-name subdirectories')
    p.add_argument('--out_dir',type=Path,required=True)
    p.add_argument('--preprocess',choices=['research','evaluate'],default='research')
    p.add_argument('--precision',choices=['research_amp','fp32'],default='research_amp')
    p.add_argument('--bootstrap',action='store_true',help='Original evaluate.py image-level1000 bootstrap; distinct from patient20k audit')
    a=p.parse_args();assert a.out_dir.is_absolute();assert not a.out_dir.exists(),'Use a new output directory; never overwrite earlier evidence'
    cfg=m.verify();torch.set_num_threads(1)
    helper=EA/'audit_evaluate_metrics.py';spec=importlib.util.spec_from_file_location('frozen_metric_namespace',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    ref=EA/'evaluate_reference.py';assert m.sha(ref)==m.sha(R/'evaluate.py')==EVALUATE_SHA
    names=['compute_metrics','load_test_data','build_transform','run_inference','bootstrap_ci','print_report','save_outputs']
    namespace=dict(vars(h),torch=torch,transforms=transforms,Image=Image,time=time,classification_report=classification_report)
    tree=ast.parse(ref.read_text());nodes=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ref),'exec'),namespace)
    if a.arm=='mom_misaligned':unit=next(u for u in m.READ(S/'selected_reference.json')['units'] if u['seed']==a.seed)
    elif a.arm=='mom_only':unit=next(u for u in m.READ(S/'working_reference.json')['units'] if u['seed']==a.seed)
    elif a.arm=='plain':unit=next(u for u in m.READ(S/'baseline_reference.json')['units'] if u['seed']==a.seed)
    else:unit=m.READ(S/f'runs/fdm_misaligned_only_s{a.seed}/complete.json')
    assert m.sha(R/unit['checkpoint'])==unit['checkpoint_sha256']
    model=m.model_for(a.seed,a.arm).eval();state=torch.load(R/unit['checkpoint'],map_location='cpu',weights_only=True)
    (model.backbone if a.arm=='plain' else model).load_state_dict(state,strict=True);del state
    mean=[.485,.456,.406];std=[.229,.224,.225]
    if a.preprocess=='research':transform=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean,std)])
    else:transform=namespace['build_transform'](224,mean,std)
    class_names=h.NAMES;indices=None
    if a.split=='val':
        indices=cfg['indices']['val'];authority={v['index']:v for v in m.READ(R/'experiments/ac_dataset_v1/dataset_manifest.json')}
        paths=[str(R/authority[i]['ac_path']) for i in indices];y=np.array([authority[i]['label'] for i in indices])
        assert len(paths)==427 and all(m.sha(Path(p))==authority[i]['sha256'] for i,p in zip(indices,paths));dataset='registered original AC validation427'
    else:
        assert a.data_path.is_absolute() and a.data_path.is_dir()
        paths,y=namespace['load_test_data'](str(a.data_path),class_names);assert len(paths)>0 and set(y.tolist())=={0,1,2};dataset=str(a.data_path)
    class ProbabilityModel(nn.Module):
        def __init__(self,base):super().__init__();self.base=base
        def forward(self,x):
            with torch.autocast('cuda',dtype=torch.float16,enabled=a.precision=='research_amp'):out=self.base(x)
            return out.float()
    dummy=torch.zeros(1,3,224,224,device='cuda');flops=m.counted_flops(model,dummy,m.base_flops);params=sum(v.numel() for v in model.parameters());del dummy
    expected=next(b for b in m.READ(S/'preflight.json')['budgets'] if b['arm']==a.arm)
    assert params==expected['parameters'] and flops==expected['counted_flops']
    complexity=dict(macs=f'{flops/2/1e9:.4f}G',flops=f'{flops/1e9:.4f}G',params=f'{params/1e6:.4f}M',params_raw=params,macs_raw=flops//2)
    wrapped=ProbabilityModel(model).eval()
    probabilities,eff=namespace['run_inference'](wrapped,paths,transform,32,torch.device('cuda'))
    predicted=probabilities.argmax(1);metrics=namespace['compute_metrics'](y,predicted,probabilities,3,class_names)
    interval=namespace['bootstrap_ci'](y,probabilities,3,1000,42) if a.bootstrap else None
    title=f'CLE_{a.arm}_s{a.seed}_single'
    namespace['print_report'](title,len(y),metrics,class_names,complexity,eff,interval)
    namespace['save_outputs'](str(a.out_dir),title,str(R/unit['checkpoint']),dataset,metrics,interval,complexity,eff,y,predicted,probabilities,class_names,paths,(mean,std,a.preprocess),224)
    # The original save_outputs truncates probabilities to6 decimals and omits some
    # metrics in results.json. Preserve its output AND a full-precision supplement.
    with (a.out_dir/'predictions_full_precision.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image_path','index','label','pred','p0','p1','p2']);writer.writeheader()
        for j,(path,label,prob) in enumerate(zip(paths,y,probabilities)):
            writer.writerow(dict(image_path=path,index=indices[j] if indices is not None else '',label=int(label),pred=int(prob.argmax()),**{f'p{k}':float(prob[k]) for k in range(3)}))
    result=dict(passed=True,arm=a.arm,seed=a.seed,checkpoint=unit['checkpoint'],checkpoint_sha256=unit['checkpoint_sha256'],config_sha256=m.sha(S/'config.json'),script_sha256=m.sha(__file__),evaluate_sha256=EVALUATE_SHA,strict_load=True,preprocess=a.preprocess,precision=a.precision,physical_batch=32,views='single',dataset=dataset,metrics=metrics,efficiency=eff,complexity=complexity,confidence_interval=interval,CI_scope='native evaluate.py per-image1000 seed42 if enabled; not paired41patient20k research audit',test_independent_generalization_claim=False)
    (a.out_dir/'full_metrics_and_identity.json').write_text(json.dumps(clean(result),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
