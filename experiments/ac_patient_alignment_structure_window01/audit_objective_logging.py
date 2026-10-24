"""Independent CPU supplement: actual objective logs and trained auxiliary weights."""
from pathlib import Path
import csv,hashlib,json,math,os,sys,time,traceback
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
def audit_completed():
 import torch
 from qualification_model import QualifiedBackbone
 from models.optimize.ac_patient_alignment_window01.library_cle import LibraryNet
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();torch.set_num_threads(1);m.verify();results=[]
 for job in m.READ(S/'jobs.json'):
  assert m.done(job);folder=S/'runs'/job['id'];complete=m.READ(folder/'complete.json');rows=list(csv.DictReader((folder/'train_log.csv').open()));assert len(rows)==60
  expected_sha=hashlib.sha256(json.dumps(sorted(job['train'])).encode()).hexdigest();errors=[]
  for epoch,row in enumerate(rows):
   assert int(row['epoch'])==epoch and int(row['actual_train_count'])==int(row['unique_train_count'])==2077 and row['train_index_set_sha256']==expected_sha
   assert all(math.isfinite(float(row[k])) for k in ['train_loss','ce_loss','contrastive_loss','valid_anchor_fraction','mean_positive_pairs','train_seconds','loader_wait_seconds'])
   assert 0<=float(row['valid_anchor_fraction'])<=1 and 0<=float(row['mean_positive_pairs'])<=31 and float(row['contrastive_loss'])>=-1e-6
   error=abs(float(row['train_loss'])-float(row['ce_loss'])-.05*float(row['contrastive_loss']));assert error<1e-5;errors.append(error)
  assert complete['best_epoch']==max(range(60),key=lambda k:float(rows[k]['val_macro_f1']))
  m.core.set_seed(job['seed']);model=LibraryNet(QualifiedBackbone('convnext_tiny',m.Q/'pretrained'),job['arm']);initial={k:v.detach().clone() for k,v in model.projector.state_dict().items()};del model
  assert m.sha(m.R/complete['checkpoint'])==complete['checkpoint_sha256'];state=torch.load(m.R/complete['checkpoint'],map_location='cpu',weights_only=True);assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point());delta={}
  for k,v in initial.items():
   trained=state['projector.'+k];assert trained.shape==v.shape and not torch.equal(trained,v);delta[k]=float((trained-v).norm())
  results.append(dict(job=job['id'],seed=job['seed'],arm=job['arm'],epochs=60,checkpoint_sha256=complete['checkpoint_sha256'],earliest_val_best=complete['best_epoch'],source_signature=complete['signature'],train_log_sha256=m.sha(folder/'train_log.csv'),all2077_index_sha=expected_sha,max_loss_weight_identity_error=max(errors),mean_valid_anchor_fraction=sum(float(r['valid_anchor_fraction']) for r in rows)/60,mean_positive_pairs=sum(float(r['mean_positive_pairs']) for r in rows)/60,mean_ce_loss=sum(float(r['ce_loss']) for r in rows)/60,mean_contrastive_loss=sum(float(r['contrastive_loss']) for r in rows)/60,initial_to_selected_checkpoint_projector_L2=delta))
  del state,initial
 return results
def main():
 m.SAVE(S/'objective_logging_finisher_identity.json',dict(pid=os.getpid(),source_sha256=m.sha(__file__),registered_at=m.core.timestamp(),CPU_only=True,new_neural_training=0))
 while not (S/'direct_control_finisher_complete.json').exists():
  assert not (S/'bootstrap_failure.json').exists() and not (S/'direct_control_finisher_failure.json').exists(),'own pipeline fault: inspect exact evidence, no retraining'
  time.sleep(5)
 assert m.READ(S/'independent_audit.json')['passed'] and m.READ(S/'direct_control_finisher_complete.json')['passed']
 while True:
  pid=m.READ(S/'direct_control_finisher_identity.json')['pid'];p=Path('/proc')/str(pid)/'cmdline'
  if not p.exists() or b'finish_direct_controls.py' not in p.read_bytes():break
  time.sleep(2)
 results=audit_completed()
 m.SAVE(S/'independent_objective_logging_audit.json',dict(passed=True,models=12,source_sha256=m.sha(__file__),units=results,source_config_sha256=m.sha(S/'config.json'),new_neural_training=0,CLE_images=0,test_predictions=0,scope='checks actual60epoch objective identities exposure and selected trained auxiliary weights; no causal or independent-generalization claim',overall_research_complete=False))
 lines=['# 实际辅助目标与参数独立审计','','CPU读取全部12完成checkpoint和原始60epoch日志，不新增图像前向或训练。逐epoch独立核验CE+0.05aux恒等式、有限值、实际2077唯一索引SHA、有效anchor及正对数范围、最早val最佳选择；重新构造同seed公开初始化投影并确认真实最佳权重已改变。','', '|方法/seed|平均有效anchor|平均正对数|最大目标恒等误差|','|---|---:|---:|---:|']
 for r in results:lines.append(f'|{r["arm"]}/{r["seed"]}|{r["mean_valid_anchor_fraction"]:.6f}|{r["mean_positive_pairs"]:.6f}|{r["max_loss_weight_identity_error"]:.3g}|')
 lines+=['', '有效anchor是随机batch实际训练暴露，尤其within可能较少；不把日志/参数更新证据当成效果或机制因果确认。完整预测与患者条件区间在主独立审计/直接控制报告，本补充不替代其结论。']
 (S/'OBJECTIVE_LOGGING_AUDIT_ZH.md').write_text('\n'.join(lines)+'\n');m.commit('audit: independently verify all12 actual patient-objective logs and trained auxiliary projection weights')
if __name__=='__main__':
 try:main()
 except Exception as e:m.SAVE(S/'objective_logging_finisher_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
