"""Local independent proof for the new strict evaluation program, no inference."""
from pathlib import Path
from fractions import Fraction
import ast,csv,hashlib,io,json,math,os,tarfile
import numpy as np
O=Path('outputs/CLE新结构研究/固定错配真正去Mom控制/完整轮')
P='experiments/ac_focal_pairing_dropmom_final_report_window01/delivery/'
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    assert os.environ['CODEX_THREAD_ID']==json.loads(Path('work/CLE_RESEARCH_CONTROL.json').read_text(encoding='utf-8'))['owner_thread_id']
    receipt=json.loads((O/'evaluation_entry_receipt.json').read_text(encoding='utf-8'));package=(O/'strict_evaluation_entry_evidence.tar.gz').read_bytes()
    assert sha(package)==receipt['sha256'] and len(package)==receipt['bytes']
    assert receipt['parent_archive_sha256']==json.loads((O/'export_receipt.json').read_text(encoding='utf-8'))['sha256']
    with tarfile.open(fileobj=io.BytesIO(package)) as t:
        assert len(t.getnames())==len(set(t.getnames()))==receipt['files']+1
        files={m.name:t.extractfile(m).read() for m in t.getmembers()}
    read=lambda p:json.loads(files[p]);manifest=read('EVALUATION_ENTRY_MANIFEST.json')
    assert manifest['parent_archive_sha256']==receipt['parent_archive_sha256']
    for p,m in manifest['files'].items():assert sha(files[p])==m['sha256'] and len(files[p])==m['bytes']
    script=files[P+'evaluate_registered_network.py'];assert sha(script)==sha(Path('work/cle_final_delivery/evaluate_registered_network.py').read_bytes())
    proof=read(P+'evaluation_entry_verification.json');assert proof['passed'] and proof['full427_original_probability_max_error']==0 and proof['val_cases']==427 and proof['test_images']==0 and proof['new_training']==0
    assert sha(files[P+'evaluation_entry_verification.json'])==receipt['proof_sha256']
    value=read(P+'validation_smoke/full_metrics_and_identity.json');assert value['strict_load'] and value['script_sha256']==sha(script) and value['physical_batch']==32 and value['preprocess']=='research' and value['precision']=='research_amp'
    assert value['checkpoint_sha256']==proof['checkpoint_sha256'] and value['evaluate_sha256']=='e5213828f81e04baf2aa025c35dfa3a1b585efbdcf8a7cfbd9033510592b0ee7'
    rows=list(csv.DictReader(io.StringIO(files[P+'validation_smoke/predictions_full_precision.csv'].decode())))
    original=next(p for p in files if p.endswith('/validation_predictions.csv'))
    old=list(csv.DictReader(io.StringIO(files[original].decode())))
    assert len(rows)==len(old)==427 and [int(v['index']) for v in rows]==[int(v['index']) for v in old]
    assert [int(v['label']) for v in rows]==[int(v['label']) for v in old]
    probabilities=np.array([[float(v[f'p{k}']) for k in range(3)] for v in rows]);expected=np.array([[float(v[f'p{k}']) for k in range(3)] for v in old]);assert np.max(np.abs(probabilities-expected))==0
    labels=np.array([int(v['label']) for v in rows]);assert [int(v['pred']) for v in rows]==probabilities.argmax(1).tolist()
    tree=ast.parse(Path('work/verify_evaluate_metric_reaudit.py').read_text(encoding='utf-8'));node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='point_scores')
    namespace=dict(globals(),labels=labels);exec(compile(ast.Module(body=[node],type_ignores=[]),'independent_point_oracle','exec'),namespace)
    exact,mcc=namespace['point_scores'](probabilities);flat={k:float(v) for k,v in value['metrics'].items() if isinstance(v,(float,int))};names=['Benign_lesions','Malignant_lesions','body_of_stomach_normal']
    for c,name in enumerate(names):
        for k in ['precision','recall','sensitivity','specificity','f1','npv']:flat[f'class{c}.{k}']=value['metrics']['per_class'][name][k]
        flat[f'class{c}.auroc']=value['metrics']['auc_roc_per_class'][name];flat[f'class{c}.auc_pr']=value['metrics']['auc_pr_per_class'][name]
    error=max([abs(float(v)-flat[k]) for k,v in exact.items()]+[abs(mcc-flat['mcc'])]);assert error<1e-12
    out=dict(passed=True,parent_archive_sha256=receipt['parent_archive_sha256'],supplement_sha256=receipt['sha256'],files=receipt['files'],actual_weight_sha256=proof['checkpoint_sha256'],full427_probability_max_error=0.,all_metrics_independent_max_error=error,exact_evaluate_metric_function=True,strictload_and_full_precision=True,new_training=0,test_images=0,overall_research_complete=False)
    (O/'local_evaluation_entry_verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    for name in ['evaluate_registered_network.py','evaluation_entry_verification.json']:(O/name).write_bytes(files[P+name])
    print(json.dumps(out))
if __name__=='__main__':main()
