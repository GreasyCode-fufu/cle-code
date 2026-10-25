from pathlib import Path
import hashlib,json,os,socket,subprocess
R=Path('/home/login/data/cle/auto/GPT').resolve();S=R/'experiments/ac_evaluate_metric_reaudit_window01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert os.getuid()==1000 and socket.gethostname()=='Dell-T2' and R==Path('/data/cle/auto/GPT')
assert all(p.stat().st_uid==1000 for p in [S,*S.rglob('*')]) and not (S/'config.json').exists()
assert sha(S/'evaluate_reference.py')==sha(R/'evaluate.py')=='e5213828f81e04baf2aa025c35dfa3a1b585efbdcf8a7cfbd9033510592b0ee7'
policy=json.loads((S/'metric_scope_policy.json').read_text());assert policy['version']==3 and policy['metric_authority']=='/home/login/data/cle/auto/GPT/evaluate.py'
cfg=dict(owner_thread_id='01a171be-ebee-7a13-b05e-97fdb6b75587',source_sha256=sha(S/'audit_evaluate_metrics.py'),evaluate_sha256=sha(S/'evaluate_reference.py'),policy_sha256=sha(S/'metric_scope_policy.json'),protocol_sha256=sha(S/'PROTOCOL_ZH.md'),registration_sha256=sha(S/'register_audit.py'),stage='post-human-clarification CPU metric audit; existing predictions only',new_training=0,new_inference=0,overall_research_complete=False)
authority=[R/'experiments/ac_dataset_v1'/n for n in ['dataset_config.json','dataset_manifest.json','independent_dataset_audit.json']]+[R/'experiments/ac_qualification_20261024_window01'/n for n in ['config.json','selected_baseline.json','pretrained_manifest.json']]+[R/'experiments/ac_test_feedback_window01/methods_v2/qualification_convnext_tiny_A.json']
cfg['locked_authority']={str(p.relative_to(R)):sha(p) for p in authority}
(S/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');(S/'.gitignore').write_text('__pycache__/\n*.pyc\nexports/\n*.tmp\n')
git='/home/login/miniconda3/bin/git';assert not subprocess.check_output([git,'-C',str(R),'diff','--cached','--name-only'],text=True).strip()
subprocess.run([git,'-C',str(R),'add','--',str(S.relative_to(R))],check=True);subprocess.run([git,'-C',str(R),'commit','-m','audit: freeze user clarified evaluate.py metrics reaudit without new inference or training'],check=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
with (S/'audit.log').open('a',buffering=1) as log:p=subprocess.Popen(['/home/login/miniconda3/envs/sunfuxiang/bin/python',str(S/'audit_evaluate_metrics.py')],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(S/'audit_identity.json').write_text(json.dumps(dict(pid=p.pid,source_config_sha256=sha(S/'config.json'),CPU_only=True,new_training=0,new_inference=0),indent=2)+'\n');print(json.dumps(dict(pid=p.pid,config_sha256=sha(S/'config.json'))))
