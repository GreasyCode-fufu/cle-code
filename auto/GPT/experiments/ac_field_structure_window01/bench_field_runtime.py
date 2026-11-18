"""Discarded-state throughput measurement for a specified frozen field arm."""
from pathlib import Path
import argparse,os,sys,time
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import run_field_structure as f
import torch
from torch.nn import functional as F

parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True,choices=f.ARMS);parser.add_argument('--slots',type=int,required=True);parser.add_argument('--index',type=int,required=True);args=parser.parse_args()
cfg=f.verify();f.bind();model=f.model_for(45+args.index%3,args.arm);opt=f.optimizer(model);scaler=torch.amp.GradScaler('cuda',init_scale=128.)
data=f.loader(cfg['indices']['train'],True,45);it=iter(data);model.train()
def step():
    global it
    start=time.perf_counter()
    try:x,y,_,_=next(it)
    except StopIteration:it=iter(data);x,y,_,_=next(it)
    wait=time.perf_counter()-start;x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(x),y)
    assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    scaler.step(opt);scaler.update();torch.cuda.synchronize();return len(y),wait
for _ in range(4):step()
folder=S/'runtime_profiles'/args.arm;f.SAVE(folder/f'ready_{args.slots}_{args.index}.json',dict(pid=os.getpid()))
gate=folder/f'gate_{args.slots}.json';deadline=time.monotonic()+240
while not gate.exists():assert time.monotonic()<deadline;time.sleep(.05)
start=f.READ(gate)['start_unix']
while time.time()<start:time.sleep(.01)
samples=0;wait=0.;tick=time.perf_counter()
while time.time()<start+12:
    n,w=step();samples+=n;wait+=w
seconds=time.perf_counter()-tick
f.SAVE(folder/f'bench_{args.slots}_{args.index}.json',dict(samples=samples,seconds=seconds,samples_per_second=samples/seconds,loader_wait_seconds=wait,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,checkpoint_saved=False))
