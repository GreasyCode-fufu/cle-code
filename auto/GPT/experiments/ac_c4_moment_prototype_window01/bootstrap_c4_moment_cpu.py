"""Register and execute only CPU mathematical checks for the new stem."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
S=Path(__file__).resolve().parent;R=S.parents[1];E=R/'experiments/ac_patient_mechanism_evidence_window01'
sys.path.insert(0,str(E));import patient_evidence as git_guard
# Reuse the quiet-index guard inside this new process only; no data/model code is called.
git_guard.S=S
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
assert os.getuid()==1000 and S.stat().st_uid==1000
files=['c4_moment_stem.py','check_c4_moment_stem.py','bootstrap_c4_moment_cpu.py','C4_MOMENT_PROTOTYPE_PROTOCOL_ZH.md']
for name in files:
    if name.endswith('.py'):compile((S/name).read_text(),str(S/name),'exec')
config=dict(source_sha256={name:sha(S/name) for name in files},prior_C4_gate_audit_sha256=sha(R/'experiments/ac_orientation_reference_window01/numerical_gate_independent_audit.json'),
    reference_model_source_sha256=sha(R/'experiments/ac_qualification_20261024_window01/qualification_model.py'),
    planned_checks=['NumPy operator oracle','C4 mean/std spatial covariance','negative random control','finite difference gradients','zero projection/CPU RNG equality','pretrained224 integration and FLOPs'],
    device='CPU',cle_images=0,network_training=0,test_images=0,novelty_established=False)
path=S/'config.json'
if path.exists():assert json.loads(path.read_text())==config
else:save(path,config)
git_guard.commit('research: register low-cost C4 orientation-moment stem hypothesis and CPU mathematical checks')
assert all(sha(S/name)==h for name,h in config['source_sha256'].items())
code=subprocess.call(['/home/login/miniconda3/envs/sunfuxiang/bin/python','-u',str(S/'check_c4_moment_stem.py')],cwd=R,
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
save(S/'cpu_exit.json',dict(returncode=code,network_training=0))
git_guard.commit('test: retain C4 stem mathematical preflight result or exact numeric failure')
raise SystemExit(code)
