"""Post hoc complete-round learning-curve audit and immutable evidence export."""
from pathlib import Path
import csv,hashlib,io,json,os,sys,tarfile,time
import numpy as np
S=Path(__file__).resolve().parent;R=S.parents[1];F=R/'experiments/ac_field_structure_window01';M=R/'experiments/ac_module_structure_window01';T=R/'experiments/ac_test_feedback_window01'
sys.path[:0]=[str(F),str(R/'experiments/ac_patient_mechanism_evidence_window01')]
import run_field_structure as f
import patient_evidence as guard
guard.S=S
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
ARMS=['B0','W0','fov_residual','field_pool','reference','no_field','no_local_norm','no_pool'];SEEDS=[45,46,47]
TITLES=['Fixed B0: ConvNeXt-Tiny 224','Matched W0: Swin-T 384','FOV residual','Field-weighted pool','Partial convolution reference','No field support','No local normalization','No weighted pooling']

def prepare():
    assert os.getuid()==1000 and S.stat().st_uid==1000 and read(F/'independent_audit.json')['passed']
    f.verify()
    if (S/'config.json').exists():return
    units=[];locked={str(p.relative_to(R)):sha(p) for p in [F/'config.json',F/'round_results.json',F/'independent_audit.json',F/'preflight.json',R/'experiments/ac_dataset_v1/dataset_config.json',R/'experiments/ac_dataset_v1/dataset_manifest.json']}
    for arm in ARMS:
        reg=None
        if arm in ['B0','W0']:
            p=T/'methods_v2'/('qualification_convnext_tiny_A.json' if arm=='B0' else 'qualification_swin_t_B.json');reg=read(p);locked[str(p.relative_to(R))]=sha(p)
        for seed in SEEDS:
            if reg:
                u=next(u for u in reg['units'] if u['seed']==seed);comp=R/u['complete'];assert sha(comp)==u['complete_sha256'];log=comp.parent/'train_log.csv';assert sha(log)==u['train_log_sha256']
            else:comp=F/'runs'/f'field_{arm}_s{seed}'/'complete.json';log=comp.parent/'train_log.csv'
            c=read(comp);dest=S/'inputs'/arm/f's{seed}';dest.mkdir(parents=True)
            for name,p in [('complete.json',comp),('train_log.csv',log)]:
                (dest/name).write_bytes(p.read_bytes());locked[str((dest/name).relative_to(R))]=sha(dest/name)
            units.append(dict(arm=arm,seed=seed,complete=str(comp.relative_to(R)),log_sha256=sha(log),checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256']))
    save(S/'config.json',dict(created_at=guard.core.timestamp(),analysis='posthoc descriptive full learning curves; no new model selection or training',seeds=SEEDS,arms=ARMS,units=units,locked=locked,source_sha256=sha(__file__),network_updates=0,existing_models_reused=24,scope='Computational audit and complete negative-results delivery; not independent generalization confirmation'))
    (S/'.gitignore').write_text('exports/\n__pycache__/\n*.pyc\n')
    guard.commit('analysis: freeze complete24-model learning-curve audit inputs after18-model field round')

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';os.nice(10);prepare();cfg=read(S/'config.json')
    assert cfg['source_sha256']==sha(__file__) and all(sha(R/p)==h for p,h in cfg['locked'].items())
    # Keep the throughput measurements free of plotting/archive CPU/I/O work.
    deadline=time.monotonic()+900
    while not (M/'profile/selected.json').exists():
        if (M/'bootstrap_failure_v3.json').exists():raise RuntimeError('module profile failed; inspect exact module log first')
        assert time.monotonic()<deadline;time.sleep(10)
    fields=['train_loss','train_acc','val_macro_f1','val_accuracy','amp_skipped_steps','loader_wait_seconds','compute_seconds','train_samples_per_second']
    indexhash=hashlib.sha256(json.dumps(sorted(read(R/'experiments/ac_dataset_v1/dataset_config.json')['indices']['train'])).encode()).hexdigest()
    values={};stats=[]
    for unit in cfg['units']:
        arm,seed=unit['arm'],unit['seed'];folder=S/'inputs'/arm/f's{seed}';c=read(folder/'complete.json');rows=list(csv.DictReader((folder/'train_log.csv').open()))
        assert len(rows)==c['epochs_completed']==60 and [int(r['epoch']) for r in rows]==list(range(60))
        assert all(int(r['actual_train_count'])==int(r['unique_train_count'])==2077 and r['train_index_set_sha256']==indexhash for r in rows)
        matrix={k:np.array([float(r[k]) for r in rows]) for k in fields};assert all(np.isfinite(z).all() for z in matrix.values())
        best=int(np.argmax(matrix['val_macro_f1']));assert best==c['best_epoch'] and abs(matrix['val_macro_f1'][best]-c['best_select_f1'])<1e-12
        values[(arm,seed)]=matrix
        stats.append(dict(arm=arm,seed=seed,best_epoch=best,best_val_f1=float(matrix['val_macro_f1'][best]),final_val_f1=float(matrix['val_macro_f1'][-1]),final_minus_best_val_f1=float(matrix['val_macro_f1'][-1]-matrix['val_macro_f1'][best]),final_train_loss=float(matrix['train_loss'][-1]),final_train_accuracy=float(matrix['train_acc'][-1]),amp_skipped_steps=int(matrix['amp_skipped_steps'].sum()),loader_wait_seconds=float(matrix['loader_wait_seconds'].sum()),compute_seconds=float(matrix['compute_seconds'].sum()),mean_epoch_samples_per_second=float(matrix['train_samples_per_second'].mean()),checkpoint_sha256=c['checkpoint_sha256']))
    save(S/'learning_curve_summary.json',dict(units=stats,models=24,new_training=0,epochs_each=60,earliest_max_recomputed=True,all_epoch_training_indices_exact=True,finite_log_values=True))
    # Seed colours retain the same meaning in every panel; markers show selection only.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib import font_manager
    available={x.name for x in font_manager.fontManager.ttflist};font=next(x for x in ['Arial','Helvetica','DejaVu Sans'] if x in available)
    plt.rcParams.update({'font.family':font,'font.size':15,'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':2,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    colors=['#0F4D92','#42949E','#B64342'];styles=['-','--',':'];x=np.arange(60);outputs=[]
    for metric,base,ylabel in [('val_macro_f1','validation_curves','Validation macro F1 (%)'),('train_loss','training_loss_curves','Training cross-entropy')]:
        fig,axes=plt.subplots(3,3,figsize=(18,12),sharex=True,sharey=True)
        allvalues=np.concatenate([v[metric] for v in values.values()]);low=float(allvalues.min());high=float(allvalues.max())
        for i,(arm,title) in enumerate(zip(ARMS,TITLES)):
            ax=axes.flat[i]
            for seed,color,style in zip(SEEDS,colors,styles):
                z=values[(arm,seed)][metric];z=z*100 if metric=='val_macro_f1' else z;ax.plot(x,z,color=color,ls=style,lw=2,label=f'Seed {seed}')
                if metric=='val_macro_f1':
                    best=int(np.argmax(z));ax.scatter(best,z[best],facecolors='white',edgecolors=color,s=55,lw=1.6,zorder=4)
            ax.set_title(f'{chr(97+i)}. {title}',fontsize=15);ax.set_xlim(-1,60);ax.set_xticks([0,20,40,59]);ax.tick_params(labelsize=12)
            if i//3==2:ax.set_xlabel('Epoch (0-based)')
            if i%3==0:ax.set_ylabel(ylabel)
            if metric=='val_macro_f1':ax.set_ylim(max(0,100*low-1),min(100,100*high+1))
            elif low>0:ax.set_yscale('log')
        leg=axes.flat[-1];leg.set_axis_off();handles=[Line2D([0],[0],color=c,ls=s,lw=2,label=f'Seed {seed}') for seed,c,s in zip(SEEDS,colors,styles)]
        if metric=='val_macro_f1':handles.append(Line2D([0],[0],color='#4D4D4D',marker='o',markerfacecolor='white',lw=0,label='Earliest validation maximum'))
        leg.legend(handles=handles,loc='center',fontsize=15)
        leg.text(.05,.13,'All 60 epochs, all paired seeds.\nNo smoothing; no test curves.\nW0 was selected using earlier test feedback.',transform=leg.transAxes,fontsize=11,va='top')
        fig.tight_layout(pad=2)
        for ext in ['png','pdf','svg']:
            p=S/'figures'/f'{base}.{ext}';p.parent.mkdir(exist_ok=True);fig.savefig(p,dpi=300,facecolor='white');outputs.append(p)
        plt.close(fig)
    results=read(F/'round_results.json');budget=read(F/'preflight.json')['budgets']
    lines=['# 几何视场结构轮：完整结果与训练曲线','','18个新增模型已完成60epoch并通过独立审计，所有seed、有效权重与阴性结论保留。另复用原B0及同骨干W0三seed日志，共24条训练曲线，不重复训练。','', '本轮没有确认视场掩码或局部归一化的可靠新增结构收益。最佳点估计的field_pool对固定B0仅+1.2972百分点，未超过1.5；对同骨干W0差值区间跨0，不能当作已证明有效的组件。该结论仅限这些已注册设计，不否定所有CLE上下文机制。','', '|方法|seed45 test F1|seed46|seed47|均值|对B0差（百分点）|对W0差（百分点）|','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS[2:]:
        u=results['summary'][arm];b=results['comparisons'][arm+'_vs_qualification_convnext_tiny_A'];w=results['comparisons'][arm+'_vs_qualification_swin_t_B'];lines.append('|'+arm+'|'+'|'.join(f'{100*v:.4f}%' for v in u['seed_macro_f1']+[u['macro_f1']])+f'|{100*b["mean_delta"]:.4f}|{100*w["mean_delta"]:.4f}|')
    lines+=['','## 训练曲线和选择记录','','![全部val曲线](figures/validation_curves.png)','','![全部train loss](figures/training_loss_curves.png)','','图中保留每epoch原始值、所有seed，空心圆标出按原val最大值选择的最早epoch；train loss在全部值正时使用log轴，没有平滑或按test选择权重。高清PNG及可编辑SVG、矢量PDF均保存。','', '|方法|seed|最佳epoch|最佳val F1|最终val F1|最终train loss|最终train accuracy|AMP跳步总数|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for u in stats:lines.append(f'|{u["arm"]}|{u["seed"]}|{u["best_epoch"]}|{100*u["best_val_f1"]:.4f}%|{100*u["final_val_f1"]:.4f}%|{u["final_train_loss"]:.6g}|{100*u["final_train_accuracy"]:.3f}%|{u["amp_skipped_steps"]}|')
    lines+=['','fov_residual、no_local_norm、no_pool的seed47均选择epoch0；这是按val规则计算所得，不能改挑后面的test较好权重。完整训练、每epoch2077唯一允许索引、有限参数及逐图概率已独立复核。这种曲线不支持机制跨seed稳定，描述性现象不等于已确定失败的因果解释。','', '## 预算与复现','','|方法|参数|逻辑GFLOPs/图|预检训练峰值MiB|','|---|---:|---:|---:|']
    for b in budget:lines.append(f'|{b["arm"]}|{b["parameters"]:,}|{b["counted_flops"]/1e9:.6f}|{b["peak_gpu_mib"]:.2f}|')
    lines+=['','权重保留于服务器weights/ac_field_structure_window01/<job>/model_best.pth；各权重SHA、逐图预测、accuracy、恶性召回、患者等权accuracy、混淆矩阵和条件bootstrap完整在原round_results.json/evaluations。训练源码/config/jobs不改；不要重新运行已完成18项。','', '```bash','cd /home/login/data/cle/auto/GPT','/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/ac_field_structure_window01/predict_field_model.py --help','```','','模型用法按原FIELD_STRUCTURE_AND_USE_ZH.md及predict_field_model.py参数选择真实完成的arm/seed与checkpoint。该轮已经结束，新的模块研究实际入口为experiments/ac_module_structure_window01/bootstrap_module_structure_v3.py，继续研究并保持目标active。','', 'test按用户要求参与开发反馈，历史test已查看且A–C构建使用其标签；等级是模型一致性代理。条件患者bootstrap不消除选择偏差，不称独立泛化确认。成熟C4条件数值里程碑及匹配C4强参照另列，不混为本轮新结构有效性。']
    (S/'FIELD_COMPLETE_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    save(S/'curve_audit.json',dict(passed=True,models=24,epochs_each=60,earliest_val_best_matches_checkpoint=True,train_indices2077_exact_every_epoch=True,finite_logged_values=True,figure_data_from_locked_logs=True,figures={str(p.relative_to(S)):sha(p) for p in outputs},input_config_sha256=sha(S/'config.json'),full_field_independent_audit_sha256=sha(F/'independent_audit.json'),overall_research_complete=False))
    guard.commit('results: preserve full field-round curves, best-epoch audit and complete negative mechanisms')
    out=S/'exports';out.mkdir(exist_ok=True);files=[];data={}
    for root in [F,S]:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or any(x in p.parts for x in ['exports','__pycache__']) or p.suffix in ['.tmp','.lock','.pyc']:continue
            name=str(p.relative_to(R));raw=p.read_bytes();data[name]=raw;files.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    target=out/'field_complete_evidence.tar.gz'
    with tarfile.open(target,'w:gz') as tar:
        for name,raw in data.items():
            item=tarfile.TarInfo(name);item.size=len(raw);item.uid=item.gid=1000;item.uname=item.gname='login';item.mode=0o644;tar.addfile(item,io.BytesIO(raw))
        raw=json.dumps(dict(files=files,models18_complete=True,weights_stored_on_server_by_sha=True,weight_binaries_included=False),ensure_ascii=False,indent=2).encode();item=tarfile.TarInfo('EVIDENCE_MANIFEST.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
    save(out/'export_receipt.json',dict(path=str(target.relative_to(R)),sha256=sha(target),bytes=target.stat().st_size,files=len(files)));print(json.dumps(read(out/'export_receipt.json')),flush=True)
if __name__=='__main__':main()
