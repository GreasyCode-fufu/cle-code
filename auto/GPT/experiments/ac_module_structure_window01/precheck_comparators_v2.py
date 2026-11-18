"""Exercise registered cached comparison inputs before new model training finishes."""
from pathlib import Path
from fractions import Fraction
import importlib.util,json,sys
import numpy as np
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import run_module_structure as m
spec=importlib.util.spec_from_file_location('module_audit_revision2',S/'audit_module_structure_v2.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
cfg=m.verify();post=m.READ(S/'postprocessing_revision_v2.json');assert post['original_config_sha256']==m.sha(S/'config.json') and all(m.sha(S/p)==h for p,h in post['sources'].items())
a.summarize();results=m.READ(S/'round_results.json');assert len(results['summary'])==5 and not results['comparisons'],'precheck is reference-only, not newly trained candidates'
ac={r['index']:r for r in m.READ(m.R/'experiments/ac_dataset_v1/dataset_manifest.json')};ids=cfg['indices']['test'];y=np.array([ac[i]['label'] for i in ids]);g=np.array([ac[i]['group_id'] for i in ids]);checks=[]
paths={}
for name in ['qualification_convnext_tiny_A','dino_baseline']:
    units=m.READ(m.T/'methods_v2'/f'{name}.json')['units'];paths[name]=[m.T/'predictions'/u['fingerprint']/'test_predictions.csv' for u in units]
roots={'B0_C4':m.R/'experiments/ac_c4_budget_controls_window01/controls/B0','DINO_C4':m.R/'experiments/ac_c4_budget_controls_window01/controls/DINO','Swin_C4':m.R/'experiments/ac_orientation_reference_window01'}
for name,root in roots.items():paths[name]=[root/'runs'/f's{s}'/'test_predictions.csv' for s in m.SEEDS]
for name,files in paths.items():
    for seed,path in zip(m.SEEDS,files):
        rows,p=a.readpred(path);assert [int(r['index']) for r in rows]==ids and len(set(ids))==576 and len(set(g))==41
        assert [int(r['label']) for r in rows]==y.tolist() and [r['group_id'] for r in rows]==g.tolist() and [r['sample_id'] for r in rows]==[ac[i]['sample_id'] for i in ids]
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6) and [int(r['pred']) for r in rows]==p.argmax(1).tolist()
        if name in roots:
            folder=path.parent;complete=m.READ(folder/'complete.json');assert m.sha(path)==complete['files']['test_predictions.csv'] and m.sha(folder/'test_views.npy')==complete['files']['test_views.npy']
            views=np.load(folder/'test_views.npy');assert np.max(np.abs(views.mean(0)-p))<1e-6
            if name!='Swin_C4':
                plain='qualification_convnext_tiny_A' if name=='B0_C4' else 'dino_baseline';_,zero=a.readpred(paths[plain][m.SEEDS.index(seed)]);assert np.max(np.abs(views[0]-zero))<1e-6
        matrix=np.zeros((3,3),dtype=np.int64)
        for truth,guess in zip(y,p.argmax(1)):matrix[truth,guess]+=1
        exact=float(sum((Fraction(2*int(matrix[k,k]),int(matrix[k,:].sum()+matrix[:,k].sum())) if matrix[k,:].sum()+matrix[:,k].sum() else Fraction(0)) for k in range(3))/3)
        target=next(u for u in results['summary'][name]['seeds'] if u['seed']==seed);assert abs(exact-target['macro_f1'])<1e-14
        checks.append(dict(method=name,seed=seed,images=576,patients=41,probability_sha256=m.sha(path),integer_fraction_f1_verified=True,macro_f1=exact))
m.SAVE(S/'comparators_precheck_v2.json',dict(passed=True,groups=5,units=15,checks=checks,postprocessing_config_sha256=m.sha(S/'postprocessing_revision_v2.json'),source_sha256=m.sha(__file__),neural_forwards=0,new_training=0,scope='Replay of already registered adaptive-test comparison probabilities; not new efficacy evidence or independent generalization.'))
m.commit('test: replay all15 cached comparator units with C4-aware reader and independent integer F1 before neural results')
print(json.dumps(dict(passed=True,groups=5,units=15)))
