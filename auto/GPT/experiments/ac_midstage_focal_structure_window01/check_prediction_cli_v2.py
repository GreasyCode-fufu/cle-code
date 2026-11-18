"""Posthoc backend repair on the SAME first fixed validation case and SAME checkpoint/tolerance."""
from pathlib import Path
import csv,json,os,subprocess,sys,time,traceback
import numpy as np
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
from verify_usage_resolution_v2 import FAILURES,verify_resolution
U=S/'usage_smoke'

def main():
 assert os.getuid()==1000 and not (U/'cli_resolution_v2.json').exists()
 cfg=m.verify();job=next(j for j in m.READ(S/'jobs.json') if j['arm']=='mom_focalcontrast' and j['seed']==45);assert m.done(job)
 comp=m.READ(S/'runs'/job['id']/'complete.json');assert comp['epochs_completed']==60 and comp['best_epoch']==2
 assert m.sha(m.R/comp['checkpoint'])==comp['checkpoint_sha256']=='79a9165a6e4863804ca1c7ef5df37584886904c86420bc5b440a28467ddf9a22'
 index=cfg['indices']['val'][0];assert index==2386
 row=next(r for r in m.READ(m.R/'experiments/ac_dataset_v1/dataset_manifest.json') if r['index']==index);image=m.R/row['ac_path'];assert m.sha(image)==row['sha256']
 failure_hashes={str(p.relative_to(m.R)):m.sha(p) for p in FAILURES}
 original=m.READ(U/'cli_prediction.json');oldbytes=(U/'cli_prediction.json').read_bytes();backup=U/'default_cli_prediction_v1.json';assert not backup.exists();backup.write_bytes(oldbytes)
 saved=list(csv.DictReader((S/'runs'/job['id']/'validation_predictions.csv').open()));expected=next(v for v in saved if int(v['index'])==index);target=np.array([float(expected[f'p{k}']) for k in range(3)])
 original_error=float(np.max(np.abs(target-np.array(original['probabilities']))));assert original_error==0.00012111663818359375 and original['checkpoint_sha256']==comp['checkpoint_sha256']
 diagnostic=m.READ(U/'native_kernel_diagnostic.json');assert diagnostic['passed'] and diagnostic['same_fixed_case']==index and diagnostic['prespecified_tolerance']==1e-4
 subprocess.run([m.PY,str(S/'predict_module_stable.py'),'--arm','mom_focalcontrast','--seed','45','--image',str(image),'--views','single','--output',str(U/'cli_prediction.json')],cwd=m.R,env=m.engine.thread_env(),stdin=subprocess.DEVNULL,check=True)
 actual=m.READ(U/'cli_prediction.json');error=float(np.max(np.abs(target-np.array(actual['probabilities']))));assert error<1e-4 and int(target.argmax())==actual['predicted_class']
 assert actual['image_sha256']==original['image_sha256']==row['sha256'] and actual['checkpoint_sha256']==original['checkpoint_sha256']
 m.SAVE(U/'independent_cli_audit.json',dict(passed=True,index=index,split='val',case_selection='first fixed registered validation index, unchanged from failedv1; not prediction-selected',image_sha256=row['sha256'],checkpoint_sha256=actual['checkpoint_sha256'],max_probability_difference_batch1_vs_original_batch32=error,prespecified_tolerance=1e-4,source_sha256=m.sha(S/'predict_module_model.py'),audit_source_sha256=m.sha(__file__),original_audit_source_sha256=m.sha(S/'check_prediction_cli.py'),wrapper_source_sha256=m.sha(S/'predict_module_stable.py'),backend_policy=actual['numerical_backend'],original_default_cli_error=original_error,posthoc_usage_numerics_repair=True,new_training=0,test_images=0,scope='Computational usage only; backend chosen after numerical diagnostics on same val case; not preregistered backend, three-seed effect, independent generalization or clinical validation'))
 repair_sources={str(p.relative_to(m.R)):m.sha(p) for p in [Path(__file__).resolve(),S/'predict_module_stable.py',S/'verify_usage_resolution_v2.py']}
 m.SAVE(U/'cli_resolution_v2.json',dict(passed=True,resolved_at=m.core.timestamp(),posthoc_usage_numerics_repair=True,index=index,prespecified_tolerance=1e-4,original_failure_error=original_error,repaired_error=error,actual_checkpoint_sha256=comp['checkpoint_sha256'],source_config_sha256=m.sha(S/'config.json'),audit_sha256=m.sha(U/'independent_cli_audit.json'),prediction_sha256=m.sha(U/'cli_prediction.json'),historical_failure_sha256=failure_hashes,repair_source_sha256=repair_sources,original_cli_prediction_sha256=m.sha(backup),diagnostic_sha256={n:m.sha(U/n) for n in ['numerics_diagnostic.json','native_kernel_diagnostic.json']},frozen_core_unchanged=True,threshold_case_weights_unchanged=True,new_training=0,test_images=0,default_CLI_remains_failed=True,native_kernel_CLI_passed=True,scope='Only resolves single-image numerical backend audit and downstream evidence exporters; no neural retraining or test-driven gain claim'))
 verify_resolution();print(json.dumps(m.READ(U/'cli_resolution_v2.json')),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:m.SAVE(S/'usage_smoke_failure_v2.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
