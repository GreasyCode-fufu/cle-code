from pathlib import Path
from fractions import Fraction
import csv,hashlib,json,sys
import numpy as np
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import texture_evidence as e
def exact_metrics(labels,pred,groups):
 cm=[[0]*3 for _ in range(3)]
 for y,p in zip(labels,pred):cm[int(y)][int(p)]+=1
 f=sum((Fraction(2*cm[k][k],sum(cm[k])+sum(cm[j][k] for j in range(3))) if sum(cm[k])+sum(cm[j][k] for j in range(3)) else Fraction() for k in range(3)),Fraction())/3
 patients=[]
 for group in sorted(set(groups)):
  subset=[i for i,g in enumerate(groups) if g==group];patients.append(Fraction(sum(int(pred[i]==labels[i]) for i in subset),len(subset)))
 return f,float(sum(patients,Fraction())/len(patients)),cm
def main():
 cfg=e.verify();result=e.read(S/'result.json');access=e.read(S/'actual_access.json');assert access['indices']==cfg['indices']['train']+cfg['indices']['val'] and access['test']==0 and len(set(access['indices']))==2504 and access['physical_cache_rows']==3595
 rows=list(csv.DictReader((S/'image_features.csv').open()));assert [int(r['index']) for r in rows]==access['indices'];features=cfg['features'];train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val'];y=np.array([int(r['label']) for r in val]);groups=np.array([r['group_id'] for r in val]);assert len(set(groups))==39
 for f in features:
  thresholds=np.quantile([float(r[f]) for r in train],[1/3,2/3],method='linear');np.testing.assert_allclose(thresholds,result['thresholds'][f],rtol=0,atol=0)
 # Direct spatial-window oracle independent of integral-image implementation.
 cache=np.load(e.R/'.cache_data_study/resized_uint8.npy',mmap_mode='r');index=cfg['indices']['train'][0];g=cache[index,16:240,16:240].mean(2)/255.;p=np.pad(g,2,mode='reflect');box=np.lib.stride_tricks.sliding_window_view(p,(5,5)).mean((-1,-2));expected=float(np.sqrt(np.mean((g-box)**2))/(g.std()+1e-4));assert abs(expected-e.features(g)['relative_fine_energy'])<1e-12
 normal=e.features(g);rotated=e.features(np.rot90(g))
 assert all(abs(normal[k]-rotated[k])<1e-10 for k in features)
 constant=e.features(np.full((224,224),.5));assert constant['std']==constant['relative_fine_energy']==constant['gradient_anisotropy']==0
 actual=e.features(g);persisted=next(r for r in train if int(r['index'])==index);assert all(abs(actual[f]-float(persisted[f]))<1e-12 for f in features)
 pred={}
 for u in cfg['units']:
  assert e.sha(e.R/u['checkpoint'])==u['checkpoint_sha256'];r=list(csv.DictReader((e.R/u['predictions']).open()));pred[(u['method'],u['seed'])]=np.array([max(range(3),key=lambda k:float(v[f'p{k}'])) for v in r])
 checked=0
 for name,slice in result['slices'].items():
  if name=='all':mask=np.ones(len(val),dtype=bool)
  else:
   feature,binname=name.rsplit('_',1);binindex=['low','mid','high'].index(binname);mask=np.searchsorted(result['thresholds'][feature],[float(r[feature]) for r in val],side='right')==binindex
  assert int(mask.sum())==slice['images'] and len(set(groups[mask]))==slice['patients'];assert np.bincount(y[mask],minlength=3).tolist()==slice['class_counts']
  for method,v in slice['methods'].items():
   for unit in v['seeds']:
    prediction=pred[(method,unit['seed'])][mask];f,patient,cm=exact_metrics(y[mask],prediction,groups[mask]);assert abs(float(f)-unit['macro_f1'])<1e-12 and abs(patient-unit['patient_equal_accuracy'])<1e-12 and cm==unit['confusion_matrix'];assert abs(float(np.mean(prediction==y[mask]))-unit['accuracy'])<1e-12;assert abs(cm[1][1]/sum(cm[1])-unit['malignant_recall'])<1e-12;checked+=1
  for key,v in slice['paired'].items():
   aa,bb=key.split('_vs_')
   for unit in v['seeds']:
    a=pred[(aa,unit['seed'])][mask]==y[mask];b=pred[(bb,unit['seed'])][mask]==y[mask];assert int(sum(a&~b))==unit['rescued'] and int(sum(~a&b))==unit['harmed']
 unique=sorted(set(groups));boot_probes=0
 for key,v in result['full_val_conditional_bootstrap'].items():
  aa,bb=key.split('_vs_');assert v['patients']==39 and v['draws']==20000
  for counts,target in zip(v['first5_draw_counts'],v['first5_delta_values']):
   duplicated=[i for group,n in zip(unique,counts) for _ in range(n) for i,gid in enumerate(groups) if gid==group];d=[]
   for seed in cfg['seeds']:
    fa,_,_=exact_metrics(y[duplicated],pred[(aa,seed)][duplicated],groups[duplicated]);fb,_,_=exact_metrics(y[duplicated],pred[(bb,seed)][duplicated],groups[duplicated]);d.append(fa-fb)
   assert abs(float(sum(d,Fraction())/3)-target)<1e-12;boot_probes+=1
 assert all((e.S/p).stat().st_uid==1000 for p in [Path('.'),*e.S.rglob('*')])
 e.save(S/'independent_audit.json',dict(passed=True,models=9,seeds=cfg['seeds'],metric_units=checked,bootstrap_direct_duplicate_oracles=boot_probes,actual_train_image_access=2077,actual_val_image_access=427,test_image_access=0,all_2504_indices_whitelisted=True,physical_cache_rows=3595,train_only_quantiles=True,source_and_actual_checkpoint_sha_verified=True,Fraction_confusion_and_patient_accuracy=True,direct_spatial_window_and_rotation_oracles=True,neural_training=0,test_predictions_read=0,independent_generalization=False));e.commit('audit: independently verify texture statistics access whitelist integer metrics and39patient bootstrap metadata')
if __name__=='__main__':main()
