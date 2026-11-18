import csv,json
from pathlib import Path
import torch

root=Path('/home/login/data/cle/auto/Network/CLE-Net')
out=root/'historical_protocol_codex_20261109'
fields=['macro_f1','accuracy','weighted_f1','balanced_accuracy','auc_roc_macro',
        'macro_sensitivity','macro_specificity','kappa','mcc']
audit=[]
for model,history,seed in [(m,h,s) for m,h in [('convnext_egb_scg','weights_cle_fixed_v1'),('convnext_tiny','weights_cleac_baseline_v2')] for s in (42,43,44)]:
    old=root/f'{history}/{model}_seed{seed}/{model}'
    new=out/f'runs/{model}_seed{seed}/{model}'
    oldeval=[]
    for p in sorted(old.glob('eval_*/results.json')):
        d=json.loads(p.read_text())
        if d.get('n_total')==576:oldeval.append((p,d))
    assert len(oldeval)==1
    op,od=oldeval[0];nd=json.loads((new/'eval_test/results.json').read_text())
    row=dict(model=model,seed=seed,historical_result=str(op),rerun_result=str(new/'eval_test/results.json'),
             metrics_differences={k:nd[k]-od[k] for k in fields},
             training_csv_exact_text_equal=(old/'train_log.csv').read_text()==(new/'train_log.csv').read_text())
    a=torch.load(old/'model_best.pth',map_location='cpu',weights_only=True)
    b=torch.load(new/'model_best.pth',map_location='cpu',weights_only=True)
    row['state_dict_keys_equal']=set(a)==set(b)
    mismatches=[];maxdiff=0.
    for k in sorted(set(a)&set(b)):
        if not torch.equal(a[k],b[k]):
            mismatches.append(k)
            if a[k].shape==b[k].shape:maxdiff=max(maxdiff,float((a[k].float()-b[k].float()).abs().max()))
    row['state_dict_tensors_count']=len(a)
    row['mismatched_tensors_count']=len(mismatches)
    row['maximum_absolute_tensor_difference']=maxdiff
    row['mismatched_keys']=mismatches
    audit.append(row)
print(json.dumps(audit,indent=2))
