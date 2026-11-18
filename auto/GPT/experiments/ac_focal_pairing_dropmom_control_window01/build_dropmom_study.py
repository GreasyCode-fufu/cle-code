from pathlib import Path
import json,hashlib,os
R=Path('/home/login/data/cle/auto/GPT').resolve();S=Path(__file__).resolve().parent;M=R/'models/optimize/ac_focal_pairing_dropmom_control_window01';old=R/'experiments/ac_midstage_focal_structure_window01'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
assert os.getuid()==1000 and S.stat().st_uid==1000 and not M.exists() and not (S/'config.json').exists()
assert os.environ['CLE_RESEARCH_OWNER']=='01a171be-ebee-7a13-b05e-97fdb6b75587'
assert read(old/'independent_audit.json')['passed'] and read(R/'experiments/ac_evaluate_metric_reaudit_window01/local_metric_reaudit_verification.json')['passed']
M.mkdir();(M/'__init__.py').write_text('');(M/'library_cle.py').write_bytes((S/'library_cle.py').read_bytes())
proto=R/'models/optimize/ac_midstage_focal_window01'
for n in ['midstage_focal.py','c4_moment_stem.py']:(M/n).write_bytes((proto/n).read_bytes())
units=[u for u in read(old/'working_reference.json')['units'] if u['arm']=='c4_moments'];assert [u['seed'] for u in units]==[45,46,47]
for u in units:assert sha(R/u['checkpoint'])==u['checkpoint_sha256'] and sha(R/u['complete'])==u['complete_sha256']
save(S/'working_reference.json',dict(units=units,selection='reuse original Mom valid three seeds; no retraining'))
reg=read(old/'evaluations/mom_misaligned/registration.json');assert reg['config_sha256']==sha(old/'config.json')
assert read(old/'evaluations/mom_misaligned/group_audit.json')['registration_sha256']==sha(old/'evaluations/mom_misaligned/registration.json')
units=reg['units'];assert [u['seed'] for u in units]==[45,46,47]
for u in units:
 u['complete']=str((old/'runs'/u['job']/'complete.json').relative_to(R))
 assert sha(R/u['checkpoint'])==u['checkpoint_sha256'] and sha(R/u['complete'])==u['complete_sha256']
save(S/'selected_reference.json',dict(units=units,original_registration_sha256=sha(old/'evaluations/mom_misaligned/registration.json'),selection='posthoc screenshot metric scope; originally fixed spatial pairing-destruction control',no_retraining=True))
save(S/'shared_snapshot_manifest.json',dict(original_focal_source_byte_exact=True,operator_sha256=sha(M/'midstage_focal.py'),original_Mom_sha256=sha(M/'c4_moment_stem.py'),new_wrapper_sha256=sha(M/'library_cle.py'),original_config_sha256=sha(old/'config.json'),shared_library_read_only=True))
save(S/'postprocessing_revision_v2.json',dict(revision=1,recipe='A original256 center224',raw_route='independent direct PIL physical3595 cache fixed val/test',loader_route='frozen T TestData A',full_val427_test576_four_views=True,registered_before_new3_training=True,adaptive_internal=True))
(S/'.gitignore').write_text('__pycache__/\n*.pyc\nexports/\npipeline.lock\nbootstrap.lock\n')
print(json.dumps(dict(study=str(S),models=str(M),new_models=3,old_weights_reused=True)))
