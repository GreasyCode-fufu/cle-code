"""Fixed first-val case, actual complete checkpoint CLI versus saved batch32 output."""
from pathlib import Path
import csv,json,os,subprocess,sys,time
import numpy as np
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
def main():
    cfg=m.verify();job=next(j for j in m.READ(S/'jobs.json') if j['arm']=='mom_odd' and j['seed']==45);
    while not m.done(job):
        assert not (S/'bootstrap_failure.json').exists(), 'own pipeline failed, inspect exact log'
        time.sleep(3)
    index=cfg['indices']['val'][0];row=next(r for r in m.READ(m.R/'experiments/ac_dataset_v1/dataset_manifest.json') if r['index']==index);image=m.R/row['ac_path'];assert m.sha(image)==row['sha256'];dest=S/'usage_smoke';dest.mkdir(exist_ok=True)
    subprocess.run([m.PY,str(S/'predict_module_model.py'),'--arm','mom_odd','--seed','45','--image',str(image),'--views','single','--output',str(dest/'cli_prediction.json')],cwd=m.R,env=m.engine.thread_env(),stdin=subprocess.DEVNULL,check=True)
    actual=m.READ(dest/'cli_prediction.json');saved=list(csv.DictReader((S/'runs'/job['id']/'validation_predictions.csv').open()));expected=next(v for v in saved if int(v['index'])==index);target=np.array([float(expected[f'p{k}']) for k in range(3)]);error=float(np.max(np.abs(target-np.array(actual['probabilities']))))
    assert error<1e-4 and int(target.argmax())==actual['predicted_class']
    m.SAVE(dest/'independent_cli_audit.json',dict(passed=True,index=index,split='val',case_selection='first fixed registered validation index, not prediction-selected',image_sha256=row['sha256'],checkpoint_sha256=actual['checkpoint_sha256'],max_probability_difference_batch1_vs_original_batch32=error,prespecified_tolerance=1e-4,source_sha256=m.sha(S/'predict_module_model.py'),audit_source_sha256=m.sha(__file__),new_training=0,test_images=0,scope='Computational usage only; not three-seed effect or clinical validation'))
    while True:
        status=m.READ(S/'status.json');safe=bool(status.get('running'))
        for j in status.get('running',[]):
            p=S/'runs'/j['job']/'train_log.csv';meta=p.parent/'run_meta.json';rows=list(csv.DictReader(p.open())) if p.exists() else [];safe &= len(rows)<45 and meta.exists() and time.time()-meta.stat().st_mtime>10
        if safe or (S/'independent_audit.json').exists():break
        time.sleep(2)
    m.commit('test: verify actual spatial odd checkpoint single-image usage against fixed first validation probability')
if __name__=='__main__':main()
