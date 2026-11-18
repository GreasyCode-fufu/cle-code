from pathlib import Path
import json,os,hashlib
R=Path('/home/login/data/cle/auto/GPT').resolve();S=Path(__file__).resolve().parent;M=R/'models/optimize/ac_merge_contrast_window01';H=R/'experiments/ac_merge_contrast_prototype_window01';P=R/'experiments/ac_harmonic_partition_structure_window01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
assert os.getuid()==1000 and S.stat().st_uid==1000 and not M.exists() and not (S/'config.json').exists()
assert read(H/'complete.json')['passed'] and all(read(P/n)['passed'] for n in ['independent_audit.json','direct_control_finisher_complete.json'])
assert not read(R/'experiments/ac_power_shape_final_report_window01/independent_summary_fraction_audit.json')['new_numerical_gate_methods']
for n in ['bootstrap_identity.json','manager_entry_identity_v2.json','runtime_sampler_identity_v2.json','direct_control_finisher_identity.json']:assert not Path('/proc',str(read(P/n)['pid']),'cmdline').exists()
mech=read(P/'mechanism_comparisons.json')['comparisons'];assert all(mech['mom_rawh2_'+v+'_vs_M_c4_moments']['mean_delta']<0 for v in ['single','c4'])
M.mkdir();(M/'__init__.py').write_text('');(M/'library_cle.py').write_bytes((S/'library_cle.py').read_bytes())
for n in ['merge_contrast.py','c4_moment_stem.py']:(M/n).write_bytes((H/n).read_bytes())
U=S/'upstream';U.mkdir()
for p in (H/'upstream').iterdir():
 if p.is_file():(U/p.name).write_bytes(p.read_bytes())
save(S/'working_reference.json',dict(selection='original M moments retained; POWER q/rawH2 not confirmed; Swin B parent test-dependent, B0 unchanged',prototype_config_sha256=sha(H/'config.json'),prototype_proof_sha256=sha(H/'numeric_preflight_cpu.json'),power_audit_sha256=sha(P/'independent_audit.json'),power_direct_sha256=sha(P/'direct_control_comparisons.json'),original_moment_source_sha256=sha(M/'c4_moment_stem.py')))
save(S/'shared_snapshot_manifest.json',dict(own_prototype_copy_exact=True,merge_source_sha256=sha(M/'merge_contrast.py'),original_Mom_sha256=sha(M/'c4_moment_stem.py'),upstream_provenance=read(U/'provenance.json'),shared_library_read_only=True))
save(S/'postprocessing_revision_v2.json',dict(revision=3,recipe='B original448 crop384',raw_route='independent direct PIL/cache positions',loader_route='frozen T TestData B',full_val427_test576_and_four_views=True,registered_before_neural_training=True,adaptive_internal=True))
print(json.dumps(dict(study=str(S),models=str(M),registered_neural_models=15)))
