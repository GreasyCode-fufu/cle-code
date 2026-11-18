"""Validate exact historical-failure lineage before accepting repaired CLI evidence."""
from pathlib import Path
import sys
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
F=m.R/'experiments/ac_midstage_focal_final_report_window01';U=S/'usage_smoke'
FAILURES=[S/'usage_smoke_failure.json',S/'usage_export_failure.json',F/'export_failure.json']

def verify_resolution():
 cfg=m.verify();r=m.READ(U/'cli_resolution_v2.json');a=m.READ(U/'independent_cli_audit.json');p=m.READ(U/'cli_prediction.json')
 assert r['passed'] and a['passed'] and r['posthoc_usage_numerics_repair']
 assert r['index']==a['index']==cfg['indices']['val'][0]==2386
 assert r['prespecified_tolerance']==a['prespecified_tolerance']==1e-4
 assert a['max_probability_difference_batch1_vs_original_batch32']<1e-4
 assert r['original_failure_error']>1e-4 and r['original_failure_error']==a['original_default_cli_error']
 assert r['actual_checkpoint_sha256']==a['checkpoint_sha256']==p['checkpoint_sha256']
 assert r['audit_sha256']==m.sha(U/'independent_cli_audit.json') and r['prediction_sha256']==m.sha(U/'cli_prediction.json')
 assert set(r['historical_failure_sha256'])=={str(f.relative_to(m.R)) for f in FAILURES}
 for rel,h in r['historical_failure_sha256'].items():assert m.sha(m.R/rel)==h
 for rel,h in r['repair_source_sha256'].items():assert m.sha(m.R/rel)==h
 assert r['source_config_sha256']==m.sha(S/'config.json')
 assert a['source_sha256']==m.sha(S/'predict_module_model.py')==cfg['sources'][str((S/'predict_module_model.py').relative_to(m.R))]
 assert a['original_audit_source_sha256']==m.sha(S/'check_prediction_cli.py')==cfg['sources'][str((S/'check_prediction_cli.py').relative_to(m.R))]
 assert a['audit_source_sha256']==m.sha(S/'check_prediction_cli_v2.py') and a['wrapper_source_sha256']==m.sha(S/'predict_module_stable.py')
 assert p['numerical_backend']['wrapper_source_sha256']==a['wrapper_source_sha256']
 assert p['numerical_backend']['forward_records']==[dict(input_shape=[1,3,224,224],cudnn_flags=dict(enabled=False,benchmark=False,deterministic=True,allow_tf32=False))]
 assert a['new_training']==a['test_images']==0 and r['new_training']==r['test_images']==0
 return r

if __name__=='__main__':print(verify_resolution())
