"""Independent local package/epoch/point/Fraction/CI verifier; numpy only."""
from pathlib import Path, PurePosixPath
from fractions import Fraction
from functools import lru_cache
import ast,csv,hashlib,io,json,math,os,tarfile
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
O=Path('outputs/CLE新结构研究/固定错配真正去Mom控制/完整轮')
S='experiments/ac_focal_pairing_dropmom_control_window01/'
F='experiments/ac_focal_pairing_dropmom_final_report_window01/'
sha=lambda x:hashlib.sha256(x).hexdigest()

def main():
    assert os.environ['CODEX_THREAD_ID']==json.loads(Path('work/CLE_RESEARCH_CONTROL.json').read_text(encoding='utf-8'))['owner_thread_id']
    receipt=json.loads((O/'export_receipt.json').read_text(encoding='utf-8'));archive=(O/'focal_dropmom_complete_evidence.tar.gz').read_bytes()
    assert sha(archive)==receipt['sha256'] and len(archive)==receipt['bytes']
    with tarfile.open(fileobj=io.BytesIO(archive)) as t:
        assert len(t.getnames())==len(set(t.getnames()))==receipt['files']+1
        assert all(m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts for m in t.getmembers())
        files={m.name:t.extractfile(m).read() for m in t.getmembers()}
    raw=lambda p:files[p];read=lambda p:json.loads(raw(p));manifest=read('EVIDENCE_MANIFEST.json')['files']
    assert len(manifest)==receipt['files']
    for p,m in manifest.items():assert sha(raw(p))==m['sha256'] and len(raw(p))==m['bytes']
    cfg=read(S+'config.json');done=read(F+'complete.json');a=read(F+'analysis_complete.json');fcfg=read(F+'config.json')
    assert sha(raw(S+'config.json'))==receipt['config_sha256']==done['training_config_sha256']=='1fbbf22a250c195195216c588ec113246fc6e05092f2ddb0717621cc25f8e781'
    assert sha(raw(F+'complete.json'))==receipt['complete_sha256'] and done['passed'] and done['models_new']==3 and done['epochs_new']==180 and done['old_effective_models_retrained']==0
    for p,h in {**cfg['sources'],**cfg['locked'],**fcfg['locked']}.items():assert sha(raw(p))==h,p
    for p,h in fcfg['sources'].items():assert sha(raw(F+p))==h,p
    assert sha(raw(S+'jobs.json'))==cfg['jobs_sha256']
    assert read(S+'module_source_audit.json')['passed'] and read(S+'matched_initialization_cpu.json')['passed']
    for b in read(S+'module_source_audit.json')['budgets']:
        assert b['initial_logits_exact'] and b['RNG_preserved'] and b['original_parameter_ids_preserved'] and b['finite_two_updates']
        for step,record in enumerate(b['gradient_records']):
            if b['arm'] in ['mom_misaligned','misaligned_only']:
                grad=record['midstage_gradients'];assert grad['up.weight']>0 and all(v==0 if step==0 else v>0 for k,v in grad.items() if k!='up.weight')
            if b['arm'].startswith('mom_'):assert record['moment_gradient']>0
    assert all(v['identical_all_retained_state'] and v['identical_fixed784_permutation'] and v['identical_rng'] and v['removed_only18528'] for v in read(S+'matched_initialization_cpu.json')['proof'])
    original='models/optimize/ac_midstage_focal_window01/';new='models/optimize/ac_focal_pairing_dropmom_control_window01/'
    for n in ['midstage_focal.py','c4_moment_stem.py']:assert raw(new+n)==raw(original+n)
    pre=read(S+'preflight.json');assert pre['passed'] and pre['test_denied'] and len(pre['reproduced_baselines'])==9
    assert all(z['full427_max_probability_error']==0 for z in pre['reproduced_baselines'])
    budget=next(v for v in pre['budgets'] if v['arm']=='misaligned_only');assert budget['parameters']==27834267 and budget['counted_flops']==8928083712 and budget['finite_two_updates']
    policy=read(S+'profile/selected.json');measure=read(S+'profile/misaligned_only/measurements.json')
    assert [v['slots'] for v in measure][:3]==[1,2,4]
    best=max(v['aggregate_samples_per_second'] for v in measure);selected=next(v['slots'] for v in measure if v['aggregate_samples_per_second']>=.97*best)
    assert policy['slots_by_arm']['misaligned_only']==selected
    cli=read(S+'usage_smoke/independent_cli_audit.json');assert cli['passed'] and cli['index']==2386 and cli['max_probability_difference_batch1_vs_original_batch32']<1e-4
    jobs=read(S+'jobs.json');assert len(jobs)==3 and [j['seed'] for j in jobs]==[45,46,47] and all(j['arm']=='misaligned_only' and j['epochs']==60 for j in jobs)
    trainhash=sha(json.dumps(sorted(cfg['indices']['train'])).encode());checkpoints=read(F+'actual_weight_bytes_reaudit.json')['weights']
    for job in jobs:
        root=S+'runs/'+job['id']+'/';c=read(root+'complete.json');log=list(csv.DictReader(io.StringIO(raw(root+'train_log.csv').decode())))
        assert len(log)==c['epochs_completed']==60 and [int(v['epoch']) for v in log]==list(range(60))
        assert c['best_epoch']==max(range(60),key=lambda n:float(log[n]['val_macro_f1']))
        assert all(int(v['actual_train_count'])==int(v['unique_train_count'])==2077 and v['train_index_set_sha256']==trainhash for v in log)
        signature=sha((json.dumps(job,sort_keys=True)+sha(raw(S+'config.json'))).encode());assert c['signature']==signature
        vc=read(root+'validation_complete.json');assert vc['signature']==signature and vc['prediction_sha256']==sha(raw(root+'validation_predictions.csv'))
        assert checkpoints[c['checkpoint']]['sha256']==c['checkpoint_sha256']
        rows=list(csv.DictReader(io.StringIO(raw(root+'validation_predictions.csv').decode())))
        assert len(rows)==427 and [int(v['index']) for v in rows]==cfg['indices']['val']
        val_authority={v['index']:v for v in read('experiments/ac_dataset_v1/dataset_manifest.json')}
        vcm=np.zeros((3,3),dtype=int)
        for row,index in zip(rows,cfg['indices']['val']):
            assert int(row['label'])==val_authority[index]['label'] and row['group_id']==val_authority[index]['group_id']
            probability=np.array([float(row[f'p{k}']) for k in range(3)]);assert np.isfinite(probability).all() and abs(probability.sum()-1)<1e-6
            vcm[int(row['label']),int(probability.argmax())]+=1
        vf1=sum((Fraction(2*int(vcm[k,k]),int(vcm[k].sum()+vcm[:,k].sum())) for k in range(3)),Fraction())/3
        assert abs(float(vf1)-c['best_select_f1'])<1e-12
    group=read(S+'evaluations/misaligned_only/group_audit.json');reg=read(S+'evaluations/misaligned_only/registration.json')
    assert group['passed'] and group['registration_sha256']==sha(raw(S+'evaluations/misaligned_only/registration.json')) and reg['config_sha256']==sha(raw(S+'config.json'))
    assert all(z['val_max_probability_error']==0 and z['test_max_probability_error']==0 for z in group['units'])
    assert read(S+'independent_audit.json')['passed'] and read(S+'independent_audit.json')['models']==3 and read(S+'bootstrap_complete.json')['passed']
    assert read(S+'runtime_sampler_complete_v2.json')['reason']=='full module round independently audited'
    manager=read(S+'manager_entry_identity_v2.json')['pid'];assert manager==read(S+'manager_identity.json')['pid']
    logs=raw(S+'bootstrap.log').decode(errors='replace');launches={}
    for line in logs.splitlines():
        if line.startswith('launch '):
            _,job,pid=line.split();assert job not in launches;launches[job]=int(pid)
    assert set(launches)=={j['id'] for j in jobs}
    traces=[json.loads(v) for v in raw(S+'continuous_runtime_process_io.jsonl').decode().splitlines()]
    for job,pid in launches.items():assert any(p['pid']==pid and p['ppid']==manager and p['job']==job for row in traces for p in row['own_processes'])
    status=read(S+'status.json');assert status['completed_new']==status['total_new']==3 and status['phase']=='module_round_audited_continue_mechanism_research'
    telemetry=list(csv.DictReader(io.StringIO(raw(S+'continuous_runtime_resources.csv').decode())))
    assert len(telemetry)>=2 and max(int(row['own_workers']) for row in telemetry)==min(selected,3)
    dt=np.diff([float(row['unix']) for row in telemetry]);assert np.all(dt>=0)
    gpu=float(np.dot(dt,[float(row['gpu_util']) for row in telemetry[:-1]])/dt.sum())
    assert not any(p.endswith(('bootstrap_failure.json','finisher_failure.json')) for p in files)
    assert a['passed'] and a['result_sha256']==sha(raw(F+'metric_results.json'))==done['result_sha256']
    assert a['input_manifest_sha256']==sha(raw(F+'input_manifest.json')) and a['weight_reaudit_sha256']==sha(raw(F+'actual_weight_bytes_reaudit.json'))
    for p,h in read(F+'input_manifest.json')['files'].items():assert sha(raw(p))==h
    result=read(F+'metric_results.json');assert result['passed'] and result['selected_historically_posthoc']
    assert result['evaluate_sha256']=='e5213828f81e04baf2aa025c35dfa3a1b585efbdcf8a7cfbd9033510592b0ee7'
    authority={v['index']:v for v in read('experiments/ac_dataset_v1/dataset_manifest.json')};ids=cfg['indices']['test'];labels=np.array([authority[i]['label'] for i in ids]);groups=[authority[i]['group_id'] for i in ids];unique=sorted(set(groups));group_ids=np.array([unique.index(g) for g in groups]);assert len(ids)==576 and len(unique)==41
    weights=np.random.default_rng(20261024).multinomial(41,np.full(41,1/41),size=20000)
    probabilities={};exacts={};points={};max_error=0.;ci_error=0.;ci_count=0
    # Reuse the already-validated independent implementation, not server helpers.
    source=Path('work/verify_evaluate_metric_reaudit.py').read_text(encoding='utf-8');tree=ast.parse(source)
    functions={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    ns=dict(globals(),labels=labels,group_ids=group_ids,weights=weights,probabilities=probabilities)
    nodes=[functions[n] for n in ['point_scores','cm_distributions','rank_distributions']]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'verified_independent_NumPy_oracle','exec'),ns)
    for name,method in result['methods'].items():
        ps=[];es=[];vs=[]
        for slot,path in enumerate(method['paths']):
            rows=list(csv.DictReader(io.StringIO(raw(path).decode())));assert len(rows)==576 and [int(z['index']) for z in rows]==ids
            assert all(int(v['label'])==authority[i]['label'] and v['group_id']==authority[i]['group_id'] for i,v in zip(ids,rows))
            prefix='' if 'p0' in rows[0] else 'mean_';p=np.array([[float(z[f'{prefix}p{k}']) for k in range(3)] for z in rows]);assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6) and p.min()>=0 and p.max()<=1
            if 'pred' in rows[0]:assert [int(z['pred']) for z in rows]==p.argmax(1).tolist()
            exact,mcc=ns['point_scores'](p);values={k:float(v) for k,v in exact.items()};values['mcc']=mcc
            assert {k:str(v) for k,v in exact.items()}==method['exact_fractions'][slot]
            assert set(values)==set(method['metrics'])
            for k,v in values.items():error=abs(v-method['metrics'][k]['seeds'][slot]);max_error=max(max_error,error);assert error<1e-12
            cm=np.zeros((3,3),dtype=int);np.add.at(cm,(labels,p.argmax(1)),1);assert cm.tolist()==method['confusion_matrices'][slot]
            ps.append(p);es.append(exact);vs.append(values)
        probabilities[name]=ps;exacts[name]=es;points[name]=vs
        for k,v in method['metrics'].items():assert abs(np.mean([z[k] for z in vs])-v['mean'])<1e-12
    for name,comparison in result['comparisons'].items():
        candidate,reference=name.split('_vs_');dist1={**ns['cm_distributions'](candidate),**ns['rank_distributions'](candidate)};dist0={**ns['cm_distributions'](reference),**ns['rank_distributions'](reference)}
        for key,value in comparison.items():
            ds=[x[key]-z[key] for x,z in zip(points[candidate],points[reference])];assert np.max(np.abs(np.array(ds)-value['paired_seed_deltas']))<1e-12
            if key in exacts[candidate][0]:
                ed=[x[key]-z[key] for x,z in zip(exacts[candidate],exacts[reference])];mean=sum(ed,Fraction())/3
                assert value['exact_mean_fraction']==str(mean) and value['exact_seed_fractions']==[str(v) for v in ed] and value['all_seeds_positive']==all(v>0 for v in ed)
                assert abs(float(mean)-value['mean_delta'])<1e-12
            else:mean=None
            if key in dist1:
                distribution=dist1[key]-dist0[key];ci=np.nanpercentile(distribution,[2.5,97.5]);error=float(np.max(np.abs(ci-value['conditional_ci95'])));ci_error=max(ci_error,error);assert error<1e-12
                assert value['valid_draws']==int(np.isfinite(distribution).sum());ci_count+=1
            qualified=bool(value['screenshot_proportional_metric'] and mean>=Fraction(3,200) and value['all_seeds_positive'] and value['conditional_ci95'][0]>0)
            assert value['conditional_numeric_qualified']==qualified
    eff=read(F+'efficiency_metrics.json');assert eff['passed'] and eff['actual_weights_strictloaded'] and eff['cases']==427 and eff['test_images']==0 and eff['physical_batch']==32 and eff['repetitions']==3
    for row in eff['results']:
        assert checkpoints[row['checkpoint']]['sha256']==row['checkpoint_sha256'] and row['MACs']*2==row['FLOPs']
        for repeat in row['runs']:
            e=repeat['efficiency'];assert e['total_samples']==427 and abs(e['pure_fps']-427/e['pure_infer_time'])<1e-9 and abs(e['e2e_fps']-427/e['e2e_time'])<1e-9
    output=dict(passed=True,archive_sha256=receipt['sha256'],files=len(manifest),new_models=3,epochs=180,original9_full427_errors_zero=True,actual_selected_cli_passed=True,own3_PID_PPID_verified=True,resource_samples=len(telemetry),observed_seconds=float(dt.sum()),timeweighted_GPU_percent=gpu,methods=len(result['methods']),complete_CSVs=3*len(result['methods']),point_max_error=max_error,conditional_intervals_checked=ci_count,conditional_CI_max_error=ci_error,pure_numpy_Fraction=True,server_actual_weight_byte_SHA_evidence=True,local_weight_bytes_downloaded=False,overall_research_complete=False)
    (O/'local_complete_verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(output))

if __name__=='__main__':main()
