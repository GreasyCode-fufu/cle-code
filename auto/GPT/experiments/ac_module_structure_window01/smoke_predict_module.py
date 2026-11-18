"""One fixed validation image: verify delivered CLI against actual completed weights."""
from pathlib import Path
import csv,json,subprocess,sys
import numpy as np
S=Path(__file__).resolve().parent;R=S.parents[1];sys.path.insert(0,str(S))
import run_module_structure as m
cfg=m.verify();job=next(j for j in m.READ(S/'jobs.json') if j['id']=='module_wtconv_s45');assert m.done(job)
i=cfg['indices']['val'][0];row=next(r for r in m.READ(R/'experiments/ac_dataset_v1/dataset_manifest.json') if r['index']==i);image=R/row['path'];assert m.sha(image)==row['sha256']
out=S/'usage_smoke/first_val_wtconv_s45.json';out.parent.mkdir(exist_ok=True)
cmd=[m.PY,str(S/'predict_module_model.py'),'--arm','wtconv','--seed','45','--image',str(image),'--views','single','--output',str(out)]
subprocess.run(cmd,cwd=R,env=m.engine.thread_env(),stdin=subprocess.DEVNULL,check=True)
pred=m.READ(out);saved=next(r for r in csv.DictReader((S/'runs'/job['id']/'validation_predictions.csv').open()) if int(r['index'])==i)
p=np.array(pred['probabilities']);target=np.array([float(saved[f'p{k}']) for k in range(3)]);error=float(np.max(np.abs(p-target)))
assert error<1e-4 and pred['predicted_class']==int(saved['pred']) and pred['checkpoint_sha256']==m.READ(S/'runs'/job['id']/'complete.json')['checkpoint_sha256']
m.SAVE(S/'usage_smoke/independent_cli_audit.json',dict(passed=True,index=i,split='val',case_selection='first fixed registered validation index, no prediction-dependent selection',image_sha256=row['sha256'],checkpoint_sha256=pred['checkpoint_sha256'],max_probability_difference_batch1_vs_original_batch32=error,source_sha256=m.sha(S/'predict_module_model.py'),audit_source_sha256=m.sha(__file__),new_training=0,test_images=0,scope='Computational model-use smoke only; not three-seed effectiveness or clinical validation.'))
m.commit('test: verify delivered module inference CLI against first actual completed checkpoint and fixed validation image')
