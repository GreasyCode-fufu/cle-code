"""Independent recomputation from locked probability arrays and per-image diagnostics."""
from pathlib import Path
from fractions import Fraction
import csv,hashlib,json,math
import numpy as np
S=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def f1cm(c):
    tp=np.diagonal(c,axis1=-2,axis2=-1);den=c.sum(-1)+c.sum(-2)
    return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0).mean(-1)
def main():
    cfg=read(S/'config.json');R=S.parents[1];assert all(sha(R/p)==h for p,h in cfg['locked'].items()) and all(sha(S/n)==h for n,h in cfg['sources'].items())
    out=read(S/'orientation_evidence_results.json');rec=read(S/'val_manifest.json');y=np.array([r['label'] for r in rec]);g=np.array([r['group_id'] for r in rec]);G=np.unique(g);assert len(y)==427 and len(G)==39
    proofs=[]
    for family in cfg['families']:
        cma=[];cmb=[]
        for seed,unit in zip(cfg['seeds'],out['summary'][family]['units']):
            views=np.load(S/'inputs'/family/f's{seed}'/'val_views.npy');old=views[0].argmax(-1);new=views.mean(0).argmax(-1);v=views.astype(float);mean=v.mean(0)
            js=(v*(np.log(np.clip(v,1e-300,1))-np.log(np.clip(mean,1e-300,1))[None])).sum(-1).mean(0)
            diag=list(csv.DictReader((S/'diagnostics'/family/f's{seed}.csv').open()));assert [int(r['index']) for r in diag]==[r['index'] for r in rec]
            assert np.max(np.abs(js-np.array([float(r['js_nats']) for r in diag])))<1e-12
            assert [int(r['original_pred']) for r in diag]==old.tolist() and [int(r['c4_pred']) for r in diag]==new.tolist()
            one=[];two=[]
            for group in G:
                a=np.zeros((3,3),dtype=int);b=np.zeros((3,3),dtype=int)
                for i in np.flatnonzero(g==group):a[y[i],new[i]]+=1;b[y[i],old[i]]+=1
                one.append(a);two.append(b)
            cma.append(one);cmb.append(two)
            for pred,target in [(old,unit['angle_metrics'][0]),(new,unit['c4_metrics'])]:
                matrix=np.zeros((3,3),dtype=int)
                for truth,guess in zip(y,pred):matrix[truth,guess]+=1
                exact=sum((Fraction(2*int(matrix[k,k]),int(matrix[k,:].sum()+matrix[:,k].sum())) if matrix[k,:].sum()+matrix[:,k].sum() else Fraction(0)) for k in range(3))/3
                assert matrix.tolist()==target['confusion_matrix'] and abs(float(exact)-target['macro_f1'])<1e-14
                assert abs(np.mean([np.mean(pred[g==group]==y[g==group]) for group in G])-target['patient_equal_accuracy'])<1e-14
            rescue=int(((old!=y)&(new==y)).sum());harm=int(((old==y)&(new!=y)).sum());assert rescue==unit['rescued_images'] and harm==unit['harmed_images']
            assert abs(np.mean([np.mean(new[g==group]==y[g==group])-np.mean(old[g==group]==y[g==group]) for group in G])-unit['patient_equal_accuracy_delta'])<1e-14
            proofs.append(dict(family=family,seed=seed,independent_fraction_confusion=True,KL_JS_identity=True,rescue_harm_identity=True,patients=39))
        # Separate explicit one-hot construction and matrix product, not the analysis einsum path.
        rng=np.random.default_rng(20261026);counts=rng.multinomial(39,np.ones(39)/39,size=20000);a=np.asarray(cma);b=np.asarray(cmb)
        delta=np.stack([f1cm((counts@a[s].reshape(39,9)).reshape(-1,3,3))-f1cm((counts@b[s].reshape(39,9)).reshape(-1,3,3)) for s in range(3)]).mean(0)
        expected=out['summary'][family]['conditional_patient_bootstrap'];assert expected['patients']==39 and expected['images']==427
        assert np.max(np.abs(np.quantile(delta,[.025,.975])-expected['ci95']))<1e-12
        # Direct duplicated-image oracle for first5 resamples, including mixed-label patients.
        for r in range(5):
            ix=np.concatenate([np.flatnonzero(g==group) for j,group in enumerate(G) for _ in range(int(counts[r,j]))]);d=[]
            for seed in cfg['seeds']:
                views=np.load(S/'inputs'/family/f's{seed}'/'val_views.npy');c=[]
                for pred in [views.mean(0).argmax(-1),views[0].argmax(-1)]:
                    mat=np.zeros((3,3),dtype=int)
                    for i in ix:mat[y[i],pred[i]]+=1
                    c.append(float(f1cm(mat)))
                d.append(c[0]-c[1])
            assert abs(float(np.mean(d))-float(delta[r]))<1e-14
    audit=dict(passed=True,probability_units=9,images_per_unit=427,patients=39,checks=proofs,full_bootstrap_independent_matrix_path=True,first5_resamples_direct_image_duplication=True,source_config_sha256=sha(S/'config.json'),result_sha256=sha(S/'orientation_evidence_results.json'),diagnostic_csv_sha256={str(p.relative_to(S)):sha(p) for p in (S/'diagnostics').rglob('*.csv')},network_training=0,network_forwards=0,test_prediction_arrays_read=0,independent_generalization_confirmation=False)
    (S/'independent_computational_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:audit[k] for k in ['passed','probability_units','images_per_unit','patients']}))
if __name__=='__main__':main()
