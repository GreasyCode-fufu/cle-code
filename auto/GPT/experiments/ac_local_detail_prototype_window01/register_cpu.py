from pathlib import Path
import hashlib
import json
import os
import socket
import subprocess
import sys

R = Path('/home/login/data/cle/auto/GPT').resolve()
S = R/'experiments/ac_local_detail_prototype_window01'
M = R/'models/optimize/ac_local_detail_prototype_window01'
P = R/'experiments/ac_midstage_focal_structure_window01'
F = R/'experiments/ac_midstage_focal_final_report_window01'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
save = lambda p, v: p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n')
owner = '01a171be-ebee-7a13-b05e-97fdb6b75587'
assert sys.argv[1:] == [owner] and os.getuid() == 1000 and socket.gethostname() == 'Dell-T2' and R == Path('/data/cle/auto/GPT')
assert all(p.stat().st_uid == 1000 for root in [S, M] for p in [root, *root.rglob('*')]) and not (S/'config.json').exists()
for name in ['independent_audit.json', 'direct_control_finisher_complete.json', 'bootstrap_complete.json']:
    assert json.loads((P/name).read_text())['passed']
assert json.loads((F/'local_complete_evidence_audit.json').read_text())['passed']
assert all(not Path('/proc', str(pid)).exists() for pid in [1167655, 1169770, 1169777, 1169778, 1176514, 1186609])
(M/'__init__.py').write_text('')
(M/'c4_moment_stem.py').write_bytes((R/'models/optimize/ac_module_window01/c4_moment_stem.py').read_bytes())
assert sha(M/'c4_moment_stem.py') == '8cd90f0659a7a852fa16c7a848bca0c416f24913c3548ee6e6141792cf429fdd'
sources = [*sorted(M.glob('*.py')), S/'check_cpu.py', S/'register_cpu.py', S/'PROTOCOL_ZH.md', *sorted((S/'references').glob('*'))]
locked = [P/name for name in ['config.json', 'independent_audit.json', 'direct_control_comparisons.json', 'bootstrap_complete.json', 'MIDSTAGE_FOCAL_RETENTION_DECISION_ZH.md']] + [F/'local_complete_evidence_audit.json'] + [R/'experiments/ac_dataset_v1'/name for name in ['dataset_config.json', 'dataset_manifest.json', 'independent_dataset_audit.json']] + [R/'experiments/ac_qualification_20261024_window01'/name for name in ['config.json', 'pretrained_manifest.json']]
save(S/'config.json', dict(identity=S.name, thread=owner,
     sources={str(p.relative_to(R)): sha(p) for p in sources}, locked={str(p.relative_to(R)): sha(p) for p in locked},
     arms=['mom_detailcontrast', 'detailcontrast_only', 'mom_detailrawgate', 'mom_misaligned', 'mom_highpass'],
     stage='CPU prototype only; no CLE images or training', before_numeric_audit=True, extra_parameters=9216,
     gradient_rule='up only firstpositive secondpositive; fixed filter no learned upstream; retainedMom separately checked',
     target='2-3 complementary network designs combined versus fixedB0 >1.5pp; no single-design gate',
     scope='mature binomial lowpass reference with firststage normalized highpass local intensity contrast; no novelty/gain guarantee',
     overall_research_complete=False))
(S/'.gitignore').write_text('__pycache__/\n*.pyc\nexports/\n')
git = '/home/login/miniconda3/bin/git'
assert not subprocess.check_output([git, '-C', str(R), 'diff', '--cached', '--name-only'], text=True).strip()
subprocess.run([git, '-C', str(R), 'add', '--', str(S.relative_to(R)), str(M.relative_to(R))], check=True)
subprocess.run([git, '-C', str(R), 'commit', '-m', 'research: preregister firststage local detail CPU prototype after terminal MFC controls'], check=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
with (S/'cpu.log').open('a', buffering=1) as log:
    process = subprocess.Popen(['/home/login/miniconda3/envs/sunfuxiang/bin/python', str(S/'check_cpu.py')], cwd=R, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
save(S/'cpu_identity.json', dict(pid=process.pid, owner_thread_id=owner, source_config_sha256=sha(S/'config.json'), CPU_only=True, CLE_images=0))
print(json.dumps(dict(pid=process.pid, config_sha256=sha(S/'config.json'), neural_training=False)))
