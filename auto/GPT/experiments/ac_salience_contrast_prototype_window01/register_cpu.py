from pathlib import Path
import hashlib,json,os,socket,subprocess,sys
R=Path('/home/login/data/cle/auto/GPT').resolve();S=R/'experiments/ac_salience_contrast_prototype_window01';M=R/'models/optimize/ac_salience_contrast_prototype_window01';C=R/'experiments/ac_channel_covariance_structure_window01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
assert os.getuid()==1000 and socket.gethostname()=='Dell-T2' and R==Path('/data/cle/auto/GPT')
assert all(p.stat().st_uid==1000 for root in [S,M] for p in [root,*root.rglob('*')]) and not (S/'config.json').exists()
assert all(json.loads((C/n).read_text())['passed'] for n in ['independent_audit.json','direct_control_finisher_complete.json','bootstrap_complete.json'])
for n in ['bootstrap_identity.json','manager_entry_identity_v2.json','runtime_sampler_identity_v2.json','direct_control_finisher_identity.json']:
 x=json.loads((C/n).read_text());assert not Path('/proc',str(x['pid']),'cmdline').exists(),n
assert not Path('/proc/1136707/cmdline').exists()
(M/'__init__.py').write_text('');(M/'c4_moment_stem.py').write_bytes((R/'models/optimize/ac_module_window01/c4_moment_stem.py').read_bytes());assert sha(M/'c4_moment_stem.py')=='8cd90f0659a7a852fa16c7a848bca0c416f24913c3548ee6e6141792cf429fdd'
(C/'COVARIANCE_RETENTION_DECISION_ZH.md').write_bytes((S/'COVARIANCE_RETENTION_DECISION_ZH.md').read_bytes())
sources=[*sorted(M.glob('*.py')),S/'check_cpu.py',S/'PROTOCOL_ZH.md',S/'register_cpu.py',*sorted((S/'references').glob('*'))]
locked=[C/n for n in ['config.json','independent_audit.json','direct_control_comparisons.json','bootstrap_complete.json']]+[R/'experiments/ac_dataset_v1'/n for n in ['dataset_config.json','dataset_manifest.json','independent_dataset_audit.json']]+[R/'experiments/ac_qualification_20261024_window01'/n for n in ['config.json','pretrained_manifest.json']]
save(S/'config.json',dict(identity=S.name,thread='01a16d4e-48fe-7e22-8c4a-c6de0f4d6f76',sources={str(p.relative_to(R)):sha(p) for p in sources},locked={str(p.relative_to(R)):sha(p) for p in locked},arms=['mom_salience','salience_only','mom_weightedmean','mom_misaligned','mom_ungated'],stage='CPU prototype only; no CLE images or training',before_numeric_audit=True,extra_parameters=dict(gated=26931,ungated=14627),gradient_rule='classifier first positive; all attention upstream first zero second positive',scope='mature attention operator adapted with LN, GAP contrast and retained classifier residual; no novelty or gain guarantee',overall_research_complete=False))
(S/'.gitignore').write_text('__pycache__/\n*.pyc\nexports/\n')
allowed=[str(p.relative_to(R)) for p in [S,M,C/'COVARIANCE_RETENTION_DECISION_ZH.md']]
git='/home/login/miniconda3/bin/git';staged=subprocess.check_output([git,'-C',str(R),'diff','--cached','--name-only'],text=True).splitlines();assert not staged,staged
subprocess.run([git,'-C',str(R),'add','--',*allowed],check=True);subprocess.run([git,'-C',str(R),'commit','-m','research: retain original Mom after complete covariance controls and preregister content attention contrast CPU prototype'],check=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');log=(S/'cpu.log').open('a',buffering=1)
p=subprocess.Popen(['/home/login/miniconda3/envs/sunfuxiang/bin/python',str(S/'check_cpu.py')],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save(S/'cpu_identity.json',dict(pid=p.pid,source_config_sha256=sha(S/'config.json'),CPU_only=True,CLE_images=0));print(json.dumps(dict(pid=p.pid,config_sha256=sha(S/'config.json'),neural_training=False)))
