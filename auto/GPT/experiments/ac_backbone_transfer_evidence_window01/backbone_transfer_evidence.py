"""Read-only historical reuse and resolution evidence; no model or image forward."""
from pathlib import Path
from fractions import Fraction
import ast,csv,hashlib,json,math,os,subprocess,time
R=Path('/home/login/data/cle/auto/GPT').resolve();S=R/'experiments/ac_backbone_transfer_evidence_window01';Q=R/'experiments/ac_qualification_20261024_window01';T=R/'experiments/ac_test_feedback_window01';N=R/'experiments/ac_harmonic_partition_structure_window01'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(S/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert os.getuid()==1000 and not (S/'complete.json').exists();cfg=read(Q/'config.json');manifest={r['index']:r for r in read(R/'experiments/ac_dataset_v1/dataset_manifest.json')};stages=[]
 names=['ac_structure_20261024_window01','ac_qualification_20261024_window01','ac_field_structure_window01','ac_module_structure_window01','ac_moment_refinement_window01','ac_crossscale_structure_window01','ac_affine_geometry_structure_window01','ac_patient_alignment_structure_window01','ac_harmonic_partition_structure_window01']
 for name in names:
  root=R/'experiments'/name;c=read(root/'config.json');record=dict(study=name,config_sha256=sha(root/'config.json'),family=c.get('family'),families=c.get('families'),recipe=c.get('recipe'),arms=c.get('arms'),jobs=len(read(root/'jobs.json')) if (root/'jobs.json').exists() else None)
  path=root/'run_module_structure.py'
  if path.exists():
   source=path.read_text();node=next((n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='model_for'),None);record.update(factory_source=ast.get_source_segment(source,node) if node else None,factory_file_sha256=sha(path))
  stages.append(record)
 cache=read(Q/'cache448_meta.json');assert cache['shape']==[2504,448,448,3] and cache['test_images']==0 and set(cache['indices'])==set(cfg['indices']['train']+cfg['indices']['val']);assert cache['cohort_manifest_sha256']=='b136e3a48ff1b6442957797b4d98228dfc7cedc9abae4f456052dcfcf4f69de8'
 def metrics(path,ids):
  rows=list(csv.DictReader(path.open()));assert [int(r['index']) for r in rows]==ids;cm=[[0]*3 for _ in range(3)];by={}
  for row in rows:
   i=int(row['index']);truth=int(row['label']);assert truth==manifest[i]['label'] and row['group_id']==manifest[i]['group_id'];p=[float(row[f'p{k}']) for k in range(3)];assert all(math.isfinite(v) and v>=0 for v in p) and abs(sum(p)-1)<1e-5;guess=max(range(3),key=lambda k:p[k]);cm[truth][guess]+=1;by.setdefault(row['group_id'],[]).append(int(truth==guess))
  score=sum((Fraction(2*cm[k][k],sum(cm[k])+sum(row[k] for row in cm)) for k in range(3)),Fraction())/3
  return dict(macro_f1=float(score),exact_macro_fraction=str(score),accuracy=sum(cm[k][k] for k in range(3))/len(rows),malignant_recall=cm[1][1]/sum(cm[1]),patient_equal_accuracy=sum(sum(v)/len(v) for v in by.values())/len(by),confusion_matrix=cm,images=len(rows),patients=len(by),prediction_sha256=sha(path))
 results={};budgets=read(Q/'preflight.json')['budgets']
 for recipe in ['A','B']:
  arm='swin_t_'+recipe;parent=read(T/'methods_v2'/('qualification_'+arm+'.json'));assert [u['seed'] for u in parent['units']]==[45,46,47];units=[]
  for u in parent['units']:
   assert sha(R/u['checkpoint'])==u['checkpoint_sha256'];folder=(R/u['complete']).parent;complete=read(R/u['complete']);log=list(csv.DictReader((folder/'train_log.csv').open()));assert len(log)==60 and complete['best_epoch']==max(range(60),key=lambda k:float(log[k]['val_macro_f1']))
   val=metrics(folder/'validation_predictions.csv',cfg['indices']['val']);test=metrics(T/'predictions'/u['fingerprint']/'test_predictions.csv',cfg['indices']['test']);units.append(dict(seed=u['seed'],checkpoint=u['checkpoint'],checkpoint_sha256=u['checkpoint_sha256'],complete_sha256=sha(R/u['complete']),val=val,test=test))
  results[arm]=dict(units=units,val_mean=sum(u['val']['macro_f1'] for u in units)/3,test_mean=sum(u['test']['macro_f1'] for u in units)/3,budget=next(b for b in budgets if b['arm']==arm),selection_scope='already adaptively viewed historical test; reference reuse, no new neural inference')
 save('evidence.json',dict(passed=True,source_sha256=sha(__file__),known_window_stages=stages,parent_references=results,recipe_A=dict(resize=256,crop=224,physical_cache_images=3595,actual_train=2077),recipe_B=dict(resize=448,crop=384,physical_cache_images=2504,allowed_splits=['train','val'],test_images_in_training_cache=0,cache_metadata_sha256=sha(Q/'cache448_meta.json'),cache_sha256=cache['sha256'],transform=cache['transform'],raw_source_sha_was_verified=cache['raw_image_sha_verified']),scope='known own-window metadata/source and completed reference predictions/checkpoint hashes only; no scan of unrelated agent directories; no new images or model forward',CLE_images=0,new_neural_training=0,new_test_inference=0,baseline_B0_unchanged=True,overall_research_complete=False))
 lines=['# 主干迁移的复用与分辨率证据','','此检查只读取本窗口已知配置/工厂源码、完成基线的概率/日志和实际权重SHA，无新图像读取、神经前向或训练。固定B0继续冻结，不以新参考替换基线。','', '|现有Swin父参考|三seed val均值|三seed test均值|参数|逻辑GFLOPs|','|---|---:|---:|---:|---:|']
 for arm,v in results.items():lines.append(f'|{arm}|{100*v["val_mean"]:.5f}%|{100*v["test_mean"]:.5f}%|{v["budget"]["parameters"]:,}|{v["budget"]["counted_flops"]/1e9:.6f}|')
 lines+=['', 'A使用原3595物理256缓存中的允许索引/crop224；B使用Q已核实原图SHA的448缓存，物理只含2077train+427val共2504/crop384，test0，不能把B描述为从256缓存上采样。两套原图清单/split不变，cache元数据与所有父参考路径/SHA列于evidence.json。原Swin A/B六个完成权重/全部seed/60epoch最早val最佳/完整427val与576test整数混淆矩阵F1已独立核验，应复用而非重训。','', '已检查的M及后续Mom相关正式轮均配置ConvNeXt-Tiny；Q/T是原Swin资格基线，F的Swin是field机制。公开Swin384方向矩数值原型不能算CLE训练。这里的记录是明确列出的本窗口源/任务范围，不宣称穷尽用户其它Agent或其它目录。后续若研究方向矩跨主干迁移，应先核这些签名和任何新增记录，不能重复有效训练，也不能仅凭工厂文本就宣称整项新方法未在任何地方存在。','', '本证据为后续机制提供准备，当前POWER15模型及其完整控制继续按原注册运行，不增删候选/改配置。任何后续结构与取舍另行预注册和数值检查；原Mom有效工作信号保留，但新主干/模块能否改善未知。旧test已被查看且AC构造用标签，当前历史指标仍自适应内部开发，不称独立泛化、新外部验证或SCI新颖性证明。']
 (S/'BACKBONE_REUSE_EVIDENCE_ZH.md').write_text('\n'.join(lines)+'\n');save('complete.json',dict(passed=True,evidence_sha256=sha(S/'evidence.json'),new_training=0,new_image_forward=0,new_test_inference=0,overall_research_complete=False))
 while True:
  staged=subprocess.check_output(['/home/login/miniconda3/bin/git','-C',str(R),'diff','--cached','--name-only'],text=True).splitlines();safe=all(p.startswith(str(S.relative_to(R))+'/') for p in staged);st=read(N/'status.json')
  if st.get('running'):
   for i in st['running']:
    p=N/'runs'/i['job']/'train_log.csv';meta=p.parent/'run_meta.json';rows=list(csv.DictReader(p.open())) if p.exists() else [];safe &= len(rows)<43 and meta.exists() and time.time()-meta.stat().st_mtime>10
  elif not (N/'independent_audit.json').exists():safe=False
  if safe:break
  time.sleep(2)
 subprocess.run(['/home/login/miniconda3/bin/git','-C',str(R),'add','--',str(S.relative_to(R))],check=True);subprocess.run(['/home/login/miniconda3/bin/git','-C',str(R),'commit','-m','audit: preserve existing Swin references exact reuse hashes and true448 cache provenance'],check=True)
 print(json.dumps(dict(passed=True,parents={k:{'val_mean':v['val_mean'],'test_mean':v['test_mean']} for k,v in results.items()},new_training=0,new_image_forward=0)),flush=True)
if __name__=='__main__':main()
