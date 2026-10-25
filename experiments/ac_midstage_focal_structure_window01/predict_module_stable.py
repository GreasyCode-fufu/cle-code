"""True B1 CLI using native CUDA convolutions; frozen model/prediction code retained."""
from pathlib import Path
import json,runpy,sys
import torch
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))

def main():
 original=S/'predict_module_model.py'
 env=runpy.run_path(str(original),run_name='frozen_prediction_implementation')
 m=env['m'];m.verify();records=[];model_for=m.model_for
 def construct(seed,arm):
  model=model_for(seed,arm)
  def before_forward(module,args):
   flags=dict(enabled=torch.backends.cudnn.enabled,benchmark=torch.backends.cudnn.benchmark,deterministic=torch.backends.cudnn.deterministic,allow_tf32=torch.backends.cudnn.allow_tf32)
   assert flags==dict(enabled=False,benchmark=False,deterministic=True,allow_tf32=False)
   assert list(args[0].shape)==[1,3,224,224],'CLI must remain physical B1, without padding or repeated images'
   records.append(dict(input_shape=list(args[0].shape),cudnn_flags=flags))
  model.register_forward_pre_hook(before_forward)
  return model
 m.model_for=construct
 with torch.backends.cudnn.flags(enabled=False,benchmark=False,deterministic=True,allow_tf32=False):
  env['main']()
 output=Path(sys.argv[sys.argv.index('--output')+1]);result=m.READ(output)
 assert len(records)==(1 if result['views']=='single' else 4)
 result['numerical_backend']=dict(policy='native CUDA convolutions, cuDNN explicitly disabled; original AMP FP16 retained',forward_records=records,wrapper_source_sha256=m.sha(__file__),frozen_prediction_source_sha256=m.sha(original),posthoc_usage_numerics_repair=True,throughput_not_equated_to_training_profile=True)
 m.SAVE(output,result);print(json.dumps(dict(backend_audit=result['numerical_backend'])))

if __name__=='__main__':main()
