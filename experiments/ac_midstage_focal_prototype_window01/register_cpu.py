from pathlib import Path
import hashlib,json,os,socket,subprocess
R=Path('/home/login/data/cle/auto/GPT').resolve()
S=R/'experiments/ac_midstage_focal_prototype_window01'
M=R/'models/optimize/ac_midstage_focal_prototype_window01'
P=R/'experiments/ac_salience_contrast_structure_window01'
F=R/'experiments/ac_salience_contrast_final_report_window01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
assert os.getuid()==1000 and socket.gethostname()=='Dell-T2' and R==Path('/data/cle/auto/GPT')
assert all(p.stat().st_uid==1000 for root in [S,M] for p in [root,*root.rglob('*')]) and not (S/'config.json').exists()
for name in ['independent_audit.json','direct_control_finisher_complete.json','bootstrap_complete.json']:
 assert json.loads((P/name).read_text())['passed']
assert json.loads((F/'local_complete_evidence_audit.json').read_text())['passed']
assert all(not Path('/proc',str(pid)).exists() for pid in [1143804,1146251,1146255,1146256,1146425,1162700])
(M/'__init__.py').write_text('')
(M/'c4_moment_stem.py').write_bytes((R/'models/optimize/ac_module_window01/c4_moment_stem.py').read_bytes())
assert sha(M/'c4_moment_stem.py')=='8cd90f0659a7a852fa16c7a848bca0c416f24913c3548ee6e6141792cf429fdd'
sources=[*sorted(M.glob('*.py')),S/'check_cpu.py',S/'register_cpu.py',S/'PROTOCOL_ZH.md',*sorted((S/'references').glob('*'))]
locked=[P/name for name in ['config.json','independent_audit.json','direct_control_comparisons.json','bootstrap_complete.json','COMBINATION_TARGET_CLARIFICATION_ZH.md','SALIENCE_RETENTION_DECISION_ZH.md']]+[F/'local_complete_evidence_audit.json']+[R/'experiments/ac_dataset_v1'/name for name in ['dataset_config.json','dataset_manifest.json','independent_dataset_audit.json']]+[R/'experiments/ac_qualification_20261024_window01'/name for name in ['config.json','pretrained_manifest.json']]
save(S/'config.json',dict(identity=S.name,thread='01a16d4e-48fe-7e22-8c4a-c6de0f4d6f76',sources={str(p.relative_to(R)):sha(p) for p in sources},locked={str(p.relative_to(R)):sha(p) for p in locked},arms=['mom_focalcontrast','focalcontrast_only','mom_focalraw','mom_misaligned','mom_additive'],stage='CPU prototype only; no CLE images or training',before_numeric_audit=True,extra_parameters=11832,gradient_rule='zero up firstpositive; down/f/h/DW firstzero secondpositive; retainedMom separately checked',target='2-3 complementary network designs combined versus fixedB0 >1.5pp; no single-design gate',scope='mature Focal operator adapted to midstage centered feature modulation; no novelty or gain guarantee',overall_research_complete=False))
(S/'.gitignore').write_text('__pycache__/\n*.pyc\nexports/\n')
git='/home/login/miniconda3/bin/git'
assert not subprocess.check_output([git,'-C',str(R),'diff','--cached','--name-only'],text=True).strip()
subprocess.run([git,'-C',str(R),'add','--',str(S.relative_to(R)),str(M.relative_to(R))],check=True)
subprocess.run([git,'-C',str(R),'commit','-m','research: preregister midstage focal contrast CPU prototype after complete salience controls'],check=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
with (S/'cpu.log').open('a',buffering=1) as log:
 p=subprocess.Popen(['/home/login/miniconda3/envs/sunfuxiang/bin/python',str(S/'check_cpu.py')],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
save(S/'cpu_identity.json',dict(pid=p.pid,source_config_sha256=sha(S/'config.json'),CPU_only=True,CLE_images=0))
print(json.dumps(dict(pid=p.pid,config_sha256=sha(S/'config.json'),neural_training=False)))
