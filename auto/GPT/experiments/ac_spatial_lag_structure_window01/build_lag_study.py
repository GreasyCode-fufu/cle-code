from pathlib import Path
import json,hashlib,os
R=Path('/home/login/data/cle/auto/GPT').resolve();S=Path(__file__).resolve().parent;M=R/'models/optimize/ac_spatial_lag_window01';H=R/'experiments/ac_spatial_lag_prototype_window01';P=R/'experiments/ac_spatial_odd_moment_structure_window01'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
assert os.getuid()==1000 and S.stat().st_uid==1000 and not M.exists() and not (S/'config.json').exists()
assert read(H/'complete.json')['passed'] and sha(H/'numeric_preflight_cpu.json')==read(H/'complete.json')['proof_sha256']
assert all(read(P/n)['passed'] for n in ['independent_audit.json','direct_control_finisher_complete.json'])
for n in ['bootstrap_identity.json','manager_entry_identity_v2.json','runtime_sampler_identity_v2.json','direct_control_finisher_identity.json']:assert not Path('/proc',str(read(P/n)['pid']),'cmdline').exists()
M.mkdir();(M/'__init__.py').write_text('');(M/'library_cle.py').write_bytes((S/'library_cle.py').read_bytes())
proto=R/'models/optimize/ac_spatial_lag_prototype_window01'
for n in ['lag_correlation.py','c4_moment_stem.py']:(M/n).write_bytes((proto/n).read_bytes())
units=[u for u in read(R/'experiments/ac_harmonic_partition_structure_window01/working_reference.json')['units'] if u['arm']=='c4_moments'];assert [u['seed'] for u in units]==[45,46,47]
for u in units:assert sha(R/u['checkpoint'])==u['checkpoint_sha256'] and sha(R/u['complete'])==u['complete_sha256']
save(S/'working_reference.json',dict(units=units,selection='reuse original CNN M moments valid three seeds; complete ODD controls do not confirm late odd increment'))
save(S/'shared_snapshot_manifest.json',dict(own_prototype_copy_exact=True,prototype_config_sha256=sha(H/'config.json'),prototype_proof_sha256=sha(H/'numeric_preflight_cpu.json'),operator_sha256=sha(M/'lag_correlation.py'),original_Mom_sha256=sha(M/'c4_moment_stem.py'),shared_library_read_only=True))
save(S/'postprocessing_revision_v2.json',dict(revision=1,recipe='A original256 center224',raw_route='independent direct PIL original physical3595 cache, fixed val/test indices',loader_route='frozen T TestData A',full_val427_test576_four_views=True,registered_before_neural_training=True,adaptive_internal=True))
(S/'.gitignore').write_text('__pycache__/\n*.pyc\nexports/\npipeline.lock\n')
print(json.dumps(dict(study=str(S),models=str(M),registered_neural_models=15)))
