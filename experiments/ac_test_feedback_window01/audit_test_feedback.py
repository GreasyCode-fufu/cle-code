"""Independent raw-array transform/inference and metric reconstruction for test feedback."""
from pathlib import Path
import csv, hashlib, json, time
import numpy as np
import torch
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def state_sha(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def compute_metrics(prob,y,groups):
    pred=prob.argmax(1)
    return dict(macro_f1=float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
        accuracy=float(accuracy_score(y,pred)),malignant_recall=float(np.mean(pred[y==1]==1)),
        patient_equal_accuracy=float(np.mean([np.mean(pred[groups==g]==y[groups==g]) for g in sorted(set(groups))])),
        confusion_matrix=confusion_matrix(y,pred,labels=[0,1,2]).tolist())

def audit_unit(group,unit,dest,model):
    from torchvision.transforms import functional as F
    from PIL import Image
    start=time.perf_counter();cfg=read(HERE/'config.json');ac=read(ROOT/'experiments/ac_dataset_v1/dataset_manifest.json')
    rows={r['index']:r for r in ac if r['split']=='test'};indices=cfg['indices'];saved=list(csv.DictReader((dest/'test_predictions.csv').open()))
    assert len(saved)==576 and [int(r['index']) for r in saved]==indices
    assert len({rows[i]['group_id'] for i in indices})==41
    for r,i in zip(saved,indices):
        assert r['sample_id']==rows[i]['sample_id'] and r['group_id']==rows[i]['group_id'] and int(r['label'])==rows[i]['label']
    target=np.array([[float(r[f'p{k}']) for k in range(3)] for r in saved]);y=np.array([rows[i]['label'] for i in indices]);g=np.array([rows[i]['group_id'] for i in indices])
    assert np.isfinite(target).all() and np.allclose(target.sum(1),1,atol=1e-6)
    assert target.min()>=0 and target.max()<=1 and [int(r['pred']) for r in saved]==target.argmax(1).tolist()
    arr=np.load(ROOT/'.cache_data_study/resized_uint8.npy' if group['recipe']=='A' else HERE/'cache/test448.npy',mmap_mode='r')
    size=224 if group['recipe']=='A' else 384;before=state_sha(model);actual=[];model.eval()
    # Reconstruct preprocessing independently, without importing TestData or its loader.
    with torch.inference_mode():
        for begin in range(0,576,32):
            batch=[]
            for pos in range(begin,min(begin+32,576)):
                j=indices[pos] if group['recipe']=='A' else pos
                im=Image.fromarray(arr[j]);im=F.center_crop(im,[size,size]);x=F.to_tensor(im)
                batch.append(F.normalize(x,[.485,.456,.406],[.229,.224,.225]))
            x=torch.stack(batch).cuda()
            with torch.autocast('cuda',dtype=torch.float16):z=model(x)
            actual.append(z.float().softmax(1).cpu().numpy())
    actual=np.concatenate(actual);error=float(np.max(np.abs(actual-target)))
    assert error<1e-6,(group['method'],unit['seed'],error)
    assert state_sha(model)==before and sha(ROOT/unit['checkpoint'])==unit['checkpoint_sha256']
    complete=read(dest/'complete.json');metrics=compute_metrics(actual,y,g)
    assert metrics==complete['metrics'] and before==complete['state_sha256']
    result=dict(passed=True,fingerprint=unit['fingerprint'],checkpoint_sha256=unit['checkpoint_sha256'],
        prediction_sha256=sha(dest/'test_predictions.csv'),test_images=576,test_patients=41,indices_and_labels_exact=True,
        independent_probability_reconstruction=True,max_probability_error=error,state_sha256=before,state_unchanged=True,
        all_parameters_finite=True,strict_load=True,parameters=sum(p.numel() for p in model.parameters()),metrics=metrics,
        seconds=time.perf_counter()-start,independent_generalization_confirmed=False,
        limitation='Independent computational audit only; adaptive test selection and prior cohort filtering remain.')
    p=dest/'independent_audit.json';part=p.with_suffix('.tmp');part.write_text(json.dumps(result,indent=2,ensure_ascii=False));part.replace(p)

def conditional_bootstrap(a,b,y,groups):
    unique=sorted(set(groups));rng=np.random.default_rng(20261024)
    weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=20000);values=[]
    for collection in [a,b]:
        cm=np.array([[confusion_matrix(y[groups==g],p.argmax(1)[groups==g],labels=[0,1,2]) for g in unique] for p in collection])
        sampled=np.einsum('dg,sgij->dsij',weights,cm)
        tp=np.diagonal(sampled,axis1=-2,axis2=-1);den=sampled.sum(-1)+sampled.sum(-2)
        values.append(np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0).mean((-1,-2)))
    delta=values[0]-values[1]
    return dict(ci95=np.percentile(delta,[2.5,97.5]).tolist(),draws=20000,patients=41,paired_seeds=3,
        patient_draws_shared_across_seeds=True,scope='conditional descriptive interval after adaptive test feedback; not selection-adjusted')
