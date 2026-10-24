"""Exact rational F1 and separately resampled patient audit of fixed C4 gate."""
from pathlib import Path
from fractions import Fraction
import csv,json,sys
import numpy as np
S=Path(__file__).resolve().parent;R=S.parents[1];sys.path.insert(0,str(S))
import run_orientation_reference as m

def load(p,ids,records,mean=False):
    rows=list(csv.DictReader(p.open()));assert [int(r['index']) for r in rows]==ids
    for r,i in zip(rows,ids):assert int(r['label'])==records[i]['label'] and r['group_id']==records[i]['group_id'] and r['sample_id']==records[i]['sample_id']
    if mean:
        angles=np.array([[[float(r[f'angle{a}_p{k}']) for k in range(3)] for r in rows] for a in [0,90,180,270]])
        p=angles.mean(0);target=np.array([[float(r[f'mean_p{k}']) for k in range(3)] for r in rows]);assert np.allclose(p,target,atol=1e-12,rtol=0)
    else:p=np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows])
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6);return p.argmax(1)

def exact_f1(y,p):
    cm=np.bincount(y*3+p,minlength=9).reshape(3,3);terms=[]
    for k in range(3):
        denominator=int(cm[k].sum()+cm[:,k].sum());terms.append(Fraction(2*int(cm[k,k]),denominator) if denominator else Fraction(0))
    return sum(terms,Fraction(0))/3,cm

cfg=m.verify();report=m.READ(S/'orientation_results.json');records={r['index']:r for r in m.READ(R/'experiments/ac_dataset_v1/dataset_manifest.json')};ids=cfg['indices']['test']
y=np.array([records[i]['label'] for i in ids]);groups=[records[i]['group_id'] for i in ids];unique=sorted(set(groups));assert len(ids)==576 and len(unique)==41
prior=m.READ(R/'experiments/ac_field_structure_window01/evidence/test_feedback_snapshot.json')['summary'];B0=prior['qualification_'+cfg['primary_baseline']]
assert [u['seed'] for u in cfg['units']]==[45,46,47]==[u['seed'] for u in B0['seeds']]
c4=[];base=[];exact=[]
for unit,b in zip(cfg['units'],B0['seeds']):
    folder=S/'runs'/f's{unit["seed"]}';proof=m.READ(folder/'complete.json');assert proof['checkpoint_sha256']==unit['checkpoint_sha256'] and proof['state_unchanged']
    assert m.sha(R/unit['checkpoint'])==unit['checkpoint_sha256'] and all(m.sha(folder/name)==h for name,h in proof['files'].items())
    p=load(folder/'test_predictions.csv',ids,records,True);bp=R/'experiments/ac_test_feedback_window01/predictions'/b['fingerprint']/'test_predictions.csv'
    audit=m.READ(bp.parent/'independent_audit.json');assert audit['passed'] and m.sha(bp)==audit['prediction_sha256'] and audit['checkpoint_sha256']==b['checkpoint_sha256']
    q=load(bp,ids,records);c4.append(p);base.append(q);fa,_=exact_f1(y,p);fb,_=exact_f1(y,q);exact.append(dict(seed=unit['seed'],C4_f1=str(fa),B0_f1=str(fb),delta=str(fa-fb)))
delta=sum((Fraction(u['delta']) for u in exact),Fraction(0))/3;assert delta>Fraction(15,1000) and all(Fraction(u['delta'])>0 for u in exact)
assert abs(float(delta)-report['comparisons']['test_C4_vs_frozen_B0']['mean_delta'])<1e-12
# Different, predeclared audit seed and direct patient draws; not a new selection rule.
rng=np.random.RandomState(20261025);samples=rng.randint(0,41,size=(20000,41));weights=np.zeros((20000,41),dtype=np.int64)
np.add.at(weights,(np.repeat(np.arange(20000),41),samples.ravel()),1);assert np.all(weights.sum(1)==41)
gi=np.array([unique.index(g) for g in groups]);scores=[]
for collection in [c4,base]:
    matrices=[]
    for pred in collection:matrices.append(np.bincount(gi*9+y*3+pred,minlength=41*9).reshape(41,3,3))
    sampled=np.einsum('dg,sgij->dsij',weights,np.array(matrices));tp=np.diagonal(sampled,axis1=-2,axis2=-1);den=sampled.sum(-1)+sampled.sum(-2)
    scores.append(np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0).mean((-1,-2)))
ci=np.percentile(scores[0]-scores[1],[2.5,97.5]).tolist();registered=report['comparisons']['test_C4_vs_frozen_B0']['conditional_patient_bootstrap']['ci95']
assert ci[0]>0 and registered[0]>0
result=dict(passed=True,config_sha256=m.sha(S/'config.json'),result_sha256=m.sha(S/'orientation_results.json'),source_sha256=m.sha(__file__),
    exact_mean_delta=str(delta),mean_delta=float(delta),strict_threshold=str(Fraction(15,1000)),seed_exact_f1=exact,
    registered_CI95=registered,separate_audit_CI95=ci,separate_audit_seed=20261025,separate_draws=20000,patients=41,images=576,
    numerical_fixed_cohort_gate_passed=True,independent_generalization_confirmed=False,selection_bias_corrected=False,
    novel_network_contribution=False,inference_cost_multiplier_vs_W0=4,overall_research_complete=False,
    scope='Mature four-view inference clears the conditional fixed-test numerical gate against precommitted B0; it is not a novel single-pass network or independent confirmation.')
m.SAVE(S/'numerical_gate_independent_audit.json',result);m.commit('audit: confirm mature C4 fixed-cohort numerical gate using exact rational F1 and independent patient resampling')
print(json.dumps(result,ensure_ascii=False),flush=True)
