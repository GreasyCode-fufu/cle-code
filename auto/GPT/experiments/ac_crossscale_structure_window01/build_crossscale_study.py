from pathlib import Path
import ast,hashlib,json,os,shutil
R=Path('/home/login/data/cle/auto/GPT').resolve();P=R/'experiments/ac_moment_refinement_window01';M0=R/'experiments/ac_module_structure_window01';S=R/'experiments/ac_crossscale_structure_window01';MODELS=R/'models/optimize/ac_crossscale_structure_window01';C=R/'experiments/ac_crossscale_prototype_window01'
assert os.getuid()==1000 and R.stat().st_uid==1000 and S.is_dir() and not (S/'config.json').exists() and not MODELS.exists()
assert json.loads((P/'independent_audit.json').read_text())['passed'] and json.loads((C/'complete_v2.json').read_text())['passed']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
MODELS.mkdir();(MODELS/'__init__.py').write_text('');(MODELS/'library_cle.py').write_bytes((S/'library_cle.py').read_bytes());(MODELS/'c4_moment_stem.py').write_bytes((R/'models/optimize/ac_module_window01/c4_moment_stem.py').read_bytes())
(S/'upstream').mkdir(exist_ok=True)
for n in ['BCNN.py','BCNN_LICENSE','MPNCOV.py','MPNCOV_LICENSE','provenance.json']:
 p=C/'upstream'/n
 if p.exists():(S/'upstream'/n).write_bytes(p.read_bytes())
(S/'upstream/retained_moment_protocol.md').write_bytes((M0/'MODULE_PROTOCOL_ZH.md').read_bytes());(S/'upstream/refinement_protocol.md').write_bytes((P/'MODULE_PROTOCOL_ZH.md').read_bytes())
working=[]
for arm in ['c4_moments','c4_aug','eca']:
 for seed in [45,46,47]:
  folder=M0/'runs'/f'module_{arm}_s{seed}';c=json.loads((folder/'complete.json').read_text());assert sha(R/c['checkpoint'])==c['checkpoint_sha256'];working.append(dict(arm=arm,seed=seed,complete=str((folder/'complete.json').relative_to(R)),complete_sha256=sha(folder/'complete.json'),checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256']))
(S/'working_reference.json').write_text(json.dumps(dict(units=working,parent_config_sha256=sha(M0/'config.json'),parent_audit_sha256=sha(M0/'independent_audit.json'),refinement_config_sha256=sha(P/'config.json'),refinement_audit_sha256=sha(P/'independent_audit.json'),selection='originalMom retained; addedECA/aug not retained; adaptively selected using prior test, B0 unchanged'),indent=2))
(S/'shared_snapshot_manifest.json').write_text(json.dumps(dict(source='frozen M moment and official statistical references',moment_source_sha256=sha(MODELS/'c4_moment_stem.py'),prototype_config_sha256=sha(C/'config.json'),prototype_proof_sha256=sha(C/'numeric_preflight_cpu.json'),upstream={str(p.relative_to(S)):sha(p) for p in sorted((S/'upstream').rglob('*')) if p.is_file()}),indent=2))
(S/'postprocessing_revision_v2.json').write_text(json.dumps(dict(parent_auditor=str((M0/'audit_module_structure_v2.py').relative_to(R)),parent_auditor_sha256=sha(M0/'audit_module_structure_v2.py'),binding='independent crossscale factory; own12model audit and paired component comparisons'),indent=2))
src=(P/'run_module_structure.py').read_text().replace('ac_moment_refinement_window01','ac_crossscale_structure_window01').replace('momentrefine_','crossscale_').replace('run_moment_manager.py','run_crossscale_manager.py').replace("['plain','moments','eca_reference',*ARMS]","['plain','moments',*ARMS]")
src=src.replace("self.rotate=training and arm=='aug_moments'","self.rotate=False")
src=src.replace('freeze paired moment-mechanism ablations and two complementary refinements on unchanged B0','freeze retained-moment crossscale correlation, spatial correspondence and faithful bilinear controls')
tree=ast.parse(src);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preflight');lines=src.splitlines(keepends=True);src=''.join(lines[:node.lineno-1])+(S/'preflight_impl.py').read_text()+'\n'+''.join(lines[node.end_lineno:]);ast.parse(src);(S/'run_module_structure.py').write_text(src)
audit=(P/'audit_module_structure.py').read_text().replace('weights/ac_moment_refinement_window01','weights/ac_crossscale_structure_window01').replace('方向矩细化：累计与消融对照','跨尺度对应统计：累计与消融对照')
audit=audit.replace('mean_only/random_moments与M_c4_moments的反向差检验完整机制；moments_eca与M_moments检验ECA增量，aug_moments与M_c4_aug检验结构是否超出成熟旋转增强。','mom_corr与M_c4_moments检验跨尺度增量；corr_only检验去方向矩；mom_shuffle等参数打乱对应，mom_bilinear保留成熟非居中二阶算子。')
(S/'audit_module_structure.py').write_text(audit)
sampler=(P/'continuous_runtime_sampler_v2.py').read_text().replace('run_moment_manager.py','run_crossscale_manager.py').replace("'continuous_runtime_sampler_v2.py'","'continuous_runtime_sampler.py'")
(S/'continuous_runtime_sampler.py').write_text(sampler)
bootstrap=(P/'bootstrap_module_structure.py').read_text().replace('run_moment_manager.py','run_crossscale_manager.py').replace("R/'experiments/ac_module_structure_window01/independent_audit.json'","R/'experiments/ac_moment_refinement_window01/independent_audit.json'")
(S/'bootstrap_module_structure.py').write_text(bootstrap)
manager="""from pathlib import Path
import subprocess,sys,os
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
m.verify();m.SAVE(S/'manager_entry_identity_v2.json',dict(pid=os.getpid(),source_sha256=m.sha(__file__),started_at=m.core.timestamp()))
if not (S/'runtime_sampler_identity_v2.json').exists():
    log=(S/'runtime_sampler.log').open('a',buffering=1);subprocess.Popen([m.PY,str(S/'continuous_runtime_sampler.py')],cwd=m.R,env=m.engine.thread_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
m.run()
"""
(S/'run_crossscale_manager.py').write_text(manager)
print(json.dumps(dict(study=str(S),models=str(MODELS),new_models=12,reused_reference_models=9,all_predecessor_neural_rounds_complete=True)))
