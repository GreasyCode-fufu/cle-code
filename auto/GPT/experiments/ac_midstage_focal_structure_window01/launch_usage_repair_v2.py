"""Register and run only replacement analysis helpers; no training process modification."""
from pathlib import Path
import csv,json,os,subprocess,sys,time,traceback
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
F=m.R/'experiments/ac_midstage_focal_final_report_window01'

def main():
 assert os.getuid()==1000;cfg=m.verify();assert not (S/'usage_repair_registration_v2.json').exists()
 for pid,name in [(1169947,'check_prediction_cli.py'),(1169948,'export_midstage_focal_final.py'),(1171301,'finalize_first_usage.py')]:
  proc=Path('/proc')/str(pid)/'cmdline';assert not(proc.exists() and name.encode() in proc.read_bytes()),'old helper still live, do not duplicate'
 paths=[S/n for n in ['launch_usage_repair_v2.py','predict_module_stable.py','check_prediction_cli_v2.py','verify_usage_resolution_v2.py','finalize_first_usage_v2.py']]+[F/'export_midstage_focal_final_v2.py']
 for p in paths:assert p.stat().st_uid==1000;compile(p.read_text(),str(p),'exec')
 reg=dict(registered_at=m.core.timestamp(),source_config_sha256=m.sha(S/'config.json'),sources={str(p.relative_to(m.R)):m.sha(p) for p in paths},posthoc_usage_numerics_repair=True,original_failures_preserved=True,threshold_case_weights_and_core_unchanged=True,new_training=0,overall_research_complete=False)
 m.SAVE(S/'usage_repair_registration_v2.json',reg);m.SAVE(F/'exporter_registration_v2.json',reg)
 env=m.engine.thread_env();env.pop('CUDA_VISIBLE_DEVICES',None)
 with (S/'usage_smoke_v2.log').open('a',buffering=1) as log:
  p=subprocess.Popen([m.PY,str(S/'check_prediction_cli_v2.py')],cwd=m.R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 m.SAVE(S/'check_prediction_cli_v2_launch_identity.json',dict(pid=p.pid,source_sha256=m.sha(S/'check_prediction_cli_v2.py'),new_training=0,started_at=m.core.timestamp()))
 assert p.wait()==0,'exact CLI v2 failed; inspect original and new failure, do not retrain'
 from verify_usage_resolution_v2 import verify_resolution
 resolution=verify_resolution()
 text='# 单图预测数值修正（结果后分析修复）\n\n首真实mom_focalcontrast45、最早valbest2、固定首val2386、原权重SHA及1e-4阈值均不变。原默认cuDNN FP16 B1概率最大差0.00012111663818359375，超差记录及原CLI输出保留；未把原默认CLI称为通过。RGB256/中心224像素与缓存严格一致，FP32 B1也未解决。\n\n显式关闭cuDNN卷积后端，仍使用原冻结预测源码、AMP FP16及物理B1（不填充到B32），native CUDA卷积单图复核误差见usage_smoke/independent_cli_audit.json，仍严格小于事前1e-4且类别一致。包装器逐forward记录B1形状及实际后端设置；m.verify、原checkpoint字节SHA和全部源cfg保持。该后端是在相同固定val图数值诊断后选择，明确是结果后使用修复，不伪称事前注册。旧失败文件及其SHA、诊断和新源码由cli_resolution_v2.json绑定。训练与全概率审计后端未改变，未重新训练，也无新test图；不推广到所有模型/图像数值差均小于阈值。\n\n实际预测请使用S/predict_module_stable.py，参数与原predict_module_model.py相同，包含--arm/--seed/--image/--views/--output；仍拒绝未完成、权重SHA错误或源cfg漂移。native单图路径未重新测吞吐，不把训练cuDNN profile当作其性能。旧STRUCTURE_AND_USE及启动198包记录原始时点，原包不改；本修正覆盖其默认预测命令。\n'
 (S/'NUMERICAL_USAGE_CORRECTION_ZH.md').write_text(text,encoding='utf-8')
 doc=S/'STRUCTURE_AND_USE_ZH.md';backup=S/'usage_smoke/STRUCTURE_AND_USE_ZH_before_backend_correction.md'
 if doc.exists():
  assert not backup.exists();backup.write_bytes(doc.read_bytes());before=m.sha(doc)
  doc.write_text('> 单图数值核验已有结果后修正：使用predict_module_stable.py，见NUMERICAL_USAGE_CORRECTION_ZH.md。原默认CLI曾超出1e-4；原失败记录和启动包保留。'+chr(10)*2+doc.read_text().replace('predict_module_model.py','predict_module_stable.py'),encoding='utf-8')
  m.SAVE(S/'usage_smoke/usage_document_correction_v2.json',dict(before_sha256=before,preserved_before_sha256=m.sha(backup),after_sha256=m.sha(doc),correction_sha256=m.sha(S/'NUMERICAL_USAGE_CORRECTION_ZH.md'),posthoc=True,frozen_core_unchanged=True))
 env=m.engine.thread_env();env['CUDA_VISIBLE_DEVICES']='';helpers=[]
 for path,name in [(S/'finalize_first_usage_v2.py','usage_export_v2'),(F/'export_midstage_focal_final_v2.py','export_v2')]:
  with (path.parent/(name+'.log')).open('a',buffering=1) as log:
   p=subprocess.Popen([m.PY,str(path)],cwd=m.R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  ident=dict(pid=p.pid,source_sha256=m.sha(path),started_at=m.core.timestamp(),CPU_only=True,new_training=0)
  m.SAVE(path.parent/(path.stem+'_launch_identity.json'),ident);helpers.append(dict(path=str(path.relative_to(m.R)),**ident))
 m.SAVE(S/'usage_repair_helpers_launched_v2.json',dict(passed=True,resolution_sha256=m.sha(S/'usage_smoke/cli_resolution_v2.json'),helpers=helpers));print(json.dumps(helpers),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:m.SAVE(S/'usage_repair_launcher_failure_v2.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
