"""New-window mature-backbone qualification. No test inference entry point."""
from pathlib import Path
import argparse,collections,csv,fcntl,hashlib,inspect,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];STUDY=Path(__file__).resolve().parent
sys.path[:0]=[str(STUDY),str(ROOT/'experiments/ac_structure_20261024_window01'),str(ROOT/'experiments'),str(ROOT)]
import run_window01 as engine
import run_data_study as core
from qualification_model import SPECS,QualifiedBackbone,counted_flops
PRIOR=ROOT/'experiments/ac_structure_20261024_window01'
COHORT=ROOT/'experiments/ac_dataset_v1'
PY='/home/login/miniconda3/envs/sunfuxiang/bin/python'
FAMILIES=list(SPECS);SEEDS=[45,46,47]
RECIPES={'A':dict(resize=256,crop=224),'B':dict(resize=448,crop=384)}
ACTIVE_RECIPE='A'
SAVE=core.save_json


def commit(message):
 path=str(STUDY.relative_to(ROOT))
 staged=subprocess.check_output([core.GIT,'-C',str(ROOT),'diff','--cached','--name-only'],text=True).splitlines()
 assert all(p.startswith(path+'/') for p in staged),'another path staged; wait before committing qualification'
 subprocess.run([core.GIT,'-C',str(ROOT),'add','--',path],check=True)
 if subprocess.run([core.GIT,'-C',str(ROOT),'diff','--cached','--quiet']).returncode:
  subprocess.run([core.GIT,'-C',str(ROOT),'commit','-m',message],check=True)


def sources():
 names=['run_qualification.py','qualification_model.py','audit_qualification.py','bootstrap_qualification.py','QUALIFICATION_PROTOCOL.md']
 paths=[STUDY/n for n in names]+[PRIOR/'run_window01.py',ROOT/'experiments/run_data_study.py']
 import torchvision.models.convnext as cn
 import torchvision.models.efficientnet as ef
 import torchvision.models.swin_transformer as sw
 out={str(p.relative_to(ROOT)):core.digest(p) for p in paths}
 out.update({f'installed:{name}':core.digest(inspect.getsourcefile(mod)) for name,mod in [('convnext',cn),('efficientnet',ef),('swin',sw)]})
 return out


def verify():
 cfg=core.read_json(STUDY/'config.json');assert cfg['sources']==sources(),'qualification source drift'
 for p,h in cfg['locked'].items():assert core.digest(ROOT/p)==h,p
 assert core.digest(STUDY/'jobs.json')==cfg['jobs_sha256']
 return cfg


def assets():
 import numpy as np,torch,torchvision
 from PIL import Image
 from concurrent.futures import ThreadPoolExecutor
 assert os.getuid()==1000 and STUDY.stat().st_uid==1000
 dest=STUDY/'pretrained';dest.mkdir(exist_ok=True)
 records={}
 for family,(factory,url,prefix) in SPECS.items():
  path=dest/f'{family}.pth'
  if not path.exists():
   partial=dest/f'{family}.{os.getpid()}.partial'
   if family=='convnext_base':shutil.copyfile('/home/login/.cache/torch/hub/checkpoints/convnext_base-6075fbad.pth',partial)
   else:torch.hub.download_url_to_file(url,str(partial),hash_prefix=prefix,progress=False)
   assert core.digest(partial).startswith(prefix)
   os.replace(partial,path)
  sha=core.digest(path);assert sha.startswith(prefix)
  records[family]=dict(url=url,sha256=sha,bytes=path.stat().st_size,path=str(path.relative_to(ROOT)))
 SAVE(STUDY/'pretrained_manifest.json',dict(assets=records,torch=torch.__version__,torchvision=torchvision.__version__,sources=sources()))
 folder=STUDY/'cache448';folder.mkdir(exist_ok=True);meta=STUDY/'cache448_meta.json'
 if meta.exists():
  info=core.read_json(meta);assert core.digest(folder/'images.npy')==info['sha256'];return
 cfg=core.read_json(COHORT/'dataset_config.json');ac=core.read_json(COHORT/'dataset_manifest.json')
 allowed=cfg['indices']['train']+cfg['indices']['val'];mapping={r['index']:r for r in ac if r['split'] in ['train','val']}
 assert len(allowed)==len(mapping)==2504 and set(allowed)==set(mapping)
 part=folder/f'images.{os.getpid()}.partial.npy'
 arr=np.lib.format.open_memmap(part,mode='w+',dtype=np.uint8,shape=(2504,448,448,3))
 def load(i):
  row=mapping[i];assert row['split'] in ['train','val']
  path=ROOT/row['path'];assert core.digest(path)==row['sha256']
  with Image.open(path) as image:return np.asarray(image.convert('RGB').resize((448,448),Image.Resampling.BILINEAR))
 with ThreadPoolExecutor(max_workers=4) as pool:
  for begin in range(0,len(allowed),128):
   for k,image in enumerate(pool.map(load,allowed[begin:begin+128])):arr[begin+k]=image
   arr.flush();print('cache448',min(begin+128,len(allowed)),len(allowed),flush=True)
 del arr;os.replace(part,folder/'images.npy')
 SAVE(meta,dict(indices=allowed,sha256=core.digest(folder/'images.npy'),shape=[2504,448,448,3],
  allowed_splits=['train','val'],raw_image_sha_verified=True,test_images=0,transform='PIL RGB bilinear resize448',
  cohort_manifest_sha256=core.digest(COHORT/'dataset_manifest.json')))


def prepare():
 if (STUDY/'config.json').exists():verify();return
 audit=core.read_json(PRIOR/'independent_audit.json');assert audit['passed']
 pairs=core.read_json(PRIOR/'paired_comparisons.json')
 assert not pairs['relational_vs_baseline']['strong_dev_signal'],'DINO signal passed; use primary research lane instead'
 data=core.read_json(COHORT/'dataset_config.json');ac=core.read_json(COHORT/'dataset_manifest.json')
 assert core.digest(COHORT/'dataset_manifest.json')==data['manifest_sha256']
 groups={s:{r['group_id'] for r in ac if r['split']==s} for s in ['train','val','test']}
 assert [len(groups[s]) for s in ['train','val','test']]==[182,39,41]
 assert all(not groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')])
 (STUDY/'manifest.json').write_bytes((COHORT/'manifest.json').read_bytes())
 refs=[]
 for seed in SEEDS:
  job=next(j for j in core.read_json(PRIOR/'jobs.json') if j['arm']=='convnext' and j['seed']==seed)
  source=PRIOR/'runs'/job['id'];comp=core.read_json(source/'complete.json')
  sig=hashlib.sha256((json.dumps(job,sort_keys=True)+core.digest(PRIOR/'config.json')).encode()).hexdigest()
  assert comp['signature']==sig and core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']
  target=STUDY/'references'/f'convnext_base_A_s{seed}';target.mkdir(parents=True,exist_ok=True)
  for name in ['complete.json','run_meta.json','train_log.csv','validation_complete.json','validation_predictions.csv']:
   (target/name).write_bytes((source/name).read_bytes())
  refs.append(dict(arm='convnext_base_A',family='convnext_base',recipe='A',seed=seed,source_signature=sig,
   checkpoint=comp['checkpoint'],checkpoint_sha256=comp['checkpoint_sha256'],
   metadata_sha256={p.name:core.digest(p) for p in target.iterdir()}))
 jobs=[]
 for family in FAMILIES:
  for recipe in RECIPES:
   if family=='convnext_base' and recipe=='A':continue
   for seed in SEEDS:
    jobs.append(dict(id=f'qualification_{family}_{recipe}_s{seed}',family=family,recipe=recipe,arm=family+'_'+recipe,
     seed=seed,epochs=60,train=data['indices']['train'],select=data['indices']['val'],score=data['indices']['val']))
 SAVE(STUDY/'jobs.json',jobs)
 locked=[COHORT/'dataset_config.json',COHORT/'dataset_manifest.json',COHORT/'independent_dataset_audit.json',
  STUDY/'manifest.json',STUDY/'pretrained_manifest.json',STUDY/'cache448_meta.json',PRIOR/'config.json',PRIOR/'independent_audit.json',
  PRIOR/'paired_comparisons.json',ROOT/'.cache_data_study/meta.json']
 import torch,torchvision
 SAVE(STUDY/'config.json',dict(identity=STUDY.name,created_at=core.timestamp(),sources=sources(),
  locked={str(p.relative_to(ROOT)):core.digest(p) for p in locked},jobs_sha256=core.digest(STUDY/'jobs.json'),
  indices=data['indices'],references=refs,families=FAMILIES,recipes=RECIPES,seeds=SEEDS,new_trainings=21,effective_models=24,
  recipe=dict(epochs=60,batch_size=32,encoder_lr=1e-5,head_lr=1e-4,weight_decay=.05,clip1=True,amp='dynamic_fp16',
    optimizer='AdamW',scheduler='cosine_to_zero',stochastic_depth=.1,efficientnet_dropout=.2,workers=2,cpu_threads=1),
  selection=dict(minimum_mean_macro_f1=.973445,minimum_seed_macro_f1=.96,tie_tolerance=.0001,
   rule='best mean among qualified; if within1e-4 choose lower FLOPs then parameters then family/recipe name'),
  trigger='Round1 audited and relational did not pass frozen development strong-signal gate',
  runtime=dict(torch=torch.__version__,torchvision=torchvision.__version__),test_inference_enabled=False))
 commit('research: freeze equal-budget qualification of four mature CLE backbones after DINO round')


def model_for(seed,arm):
 import torch
 family,recipe=arm.rsplit('_',1);assert recipe in RECIPES
 info=core.read_json(STUDY/'pretrained_manifest.json')['assets'][family]
 assert core.digest(ROOT/info['path'])==info['sha256'],'pretrained asset drift'
 core.set_seed(seed);torch.set_num_threads(1)
 return QualifiedBackbone(family,STUDY/'pretrained').cuda()


class GuardedData:
 def __init__(self,indices,training,recipe):
  import numpy as np
  from torchvision import transforms as T
  cfg=core.read_json(STUDY/'config.json');allowed=cfg['indices']['train' if training else 'val']
  assert len(indices)==len(set(indices)) and set(indices)<=set(allowed),'unauthorized split access'
  self.indices=list(indices);self.records=core.read_json(STUDY/'manifest.json')
  self.recipe=recipe
  if recipe=='A':
   self.array=np.load(ROOT/'.cache_data_study/resized_uint8.npy',mmap_mode='r');self.positions=None
   assert self.array.shape==(3595,256,256,3)
  else:
   m=core.read_json(STUDY/'cache448_meta.json');self.positions={i:p for p,i in enumerate(m['indices'])}
   self.array=np.load(STUDY/'cache448/images.npy',mmap_mode='r');assert self.array.shape==(2504,448,448,3)
  size=RECIPES[recipe]['crop']
  transforms=([T.RandomHorizontalFlip(),T.RandomVerticalFlip(),T.RandomRotation(15),T.RandomCrop(size)] if training else [T.CenterCrop(size)])
  self.transform=T.Compose(transforms+[T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 def __len__(self):return len(self.indices)
 def __getitem__(self,pos):
  from PIL import Image
  i=self.indices[pos];p=i if self.positions is None else self.positions[i]
  return self.transform(Image.fromarray(self.array[p])),self.records[i]['label'],1.,i


def loader(indices,training,seed):
 import torch
 return torch.utils.data.DataLoader(GuardedData(indices,training,ACTIVE_RECIPE),batch_size=32,shuffle=training,
  num_workers=2,pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(seed))


def signature(job):return hashlib.sha256((json.dumps(job,sort_keys=True)+core.digest(STUDY/'config.json')).encode()).hexdigest()


def done(job):
 p=STUDY/'runs'/job['id']/'validation_complete.json'
 if not p.exists():return False
 x=core.read_json(p);assert x['signature']==signature(job)
 assert core.digest(ROOT/x['checkpoint'])==x['checkpoint_sha256']
 assert core.digest(p.parent/'validation_predictions.csv')==x['prediction_sha256']
 return True


def bind():
 # Bind immutable round1 training implementation inside this NEW PROCESS only.
 # Old files and other processes retain their original globals and signatures.
 engine.STUDY=STUDY;core.STUDY=STUDY;core.WEIGHTS=ROOT/'weights/ac_qualification_20261024_window01'
 engine.verify=verify;engine.source_hashes=sources;engine.model_for=model_for;engine.loader=loader
 engine.signature=signature;engine.done=done


def worker(job_id):
 global ACTIVE_RECIPE
 job=next(j for j in core.read_json(STUDY/'jobs.json') if j['id']==job_id);ACTIVE_RECIPE=job['recipe']
 bind();engine.worker(job_id)


def preflight():
 global ACTIVE_RECIPE
 import numpy as np,torch
 from torch.nn import functional as F
 from torchvision import transforms as T
 cfg=verify();bind()
 cache=core.read_json(STUDY/'cache448_meta.json');assert core.digest(STUDY/'cache448/images.npy')==cache['sha256']
 original_cache=core.read_json(ROOT/'.cache_data_study/meta.json')
 assert core.digest(ROOT/'.cache_data_study/resized_uint8.npy')==original_cache['cache_sha256']
 budgets=[]
 for family in FAMILIES:
  initial=None
  for recipe in RECIPES:
   ACTIVE_RECIPE=recipe;model=model_for(45,family+'_'+recipe)
   state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   if initial is None:initial=state
   else:assert all(torch.equal(v,initial[k]) for k,v in state.items()),'paired resolution initial states differ'
   x,y,_,idx=next(iter(loader(cfg['indices']['train'][:32],True,45)));x,y=x.cuda(),y.cuda()
   assert x.shape==(32,3,RECIPES[recipe]['crop'],RECIPES[recipe]['crop'])
   opt=engine.optimizer(model);ids=[id(p) for g in opt.param_groups for p in g['params']]
   assert len(ids)==len(set(ids))==len(list(model.parameters()))
   assert [g['lr'] for g in opt.param_groups]==[1e-5,1e-4]
   model.eval();flops=counted_flops(model,x[:1])
   torch.cuda.reset_peak_memory_stats();model.train();scaler=torch.amp.GradScaler('cuda',init_scale=128.)
   for step in range(2):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(x),y)
    scaler.scale(loss).backward();scaler.unscale_(opt)
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
    scaler.step(opt);scaler.update()
   assert not torch.equal(model.head.weight.detach().cpu(),initial['head.weight']),'head did not update'
   changed_encoder=sum(not torch.equal(value.detach().cpu(),initial['encoder.'+name]) for name,value in model.encoder.named_parameters())
   assert changed_encoder>0,'encoder parameters did not update'
   budgets.append(dict(arm=family+'_'+recipe,family=family,recipe=recipe,parameters=sum(p.numel() for p in model.parameters()),
    counted_flops=flops,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,finite_real_two_updates=True))
   del model,opt,state,x,y,loss,norm,scaler;torch.cuda.empty_cache()
  del initial
 # Compare allowed A image tensors with immutable historical loader exactly.
 ACTIVE_RECIPE='A';a=GuardedData(cfg['indices']['val'][:8],False,'A');b=core.build_dataset(cfg['indices']['val'][:8],False)
 assert all(torch.equal(a[k][0],b[k][0]) and a[k][1:]==b[k][1:] for k in range(8))
 # Guard BOTH caches against test or D/E metadata access.
 for recipe in RECIPES:
  try:GuardedData(cfg['indices']['test'][:1],False,recipe);raise RuntimeError('test guard failure')
  except AssertionError as e:assert 'unauthorized split access' in str(e)
 for ref in cfg['references']:
  model=model_for(ref['seed'],ref['arm']);model.load_state_dict(torch.load(ROOT/ref['checkpoint'],map_location='cpu',weights_only=True),strict=True)
  p,y,idx=core.infer(model,loader(cfg['indices']['val'],False,ref['seed']))
  folder=STUDY/'references'/f'convnext_base_A_s{ref["seed"]}';saved=list(csv.DictReader((folder/'validation_predictions.csv').open()))
  target=np.array([[float(row[f'p{k}']) for k in range(3)] for row in saved])
  assert idx==cfg['indices']['val'] and np.max(np.abs(p-target))<1e-6
  del model;torch.cuda.empty_cache()
 SAVE(STUDY/'preflight.json',dict(passed=True,timestamp=core.timestamp(),budgets=budgets,
  checks=['official pretrained strict load/full SHA','paired224/384 encoder and head initial states','two finite real AMP updates',
   'exact optimizer coverage/lr','old224 transform equality','new train/val-only448 cache SHA/raw-image SHA',
   'test access rejection for both caches','all three ConvNeXtBase224 full427 probability reproduction'],
  flops_excludes='norm, activation, pooling, softmax, bias, indexing and memory; explicit Swin QKV/project plus attention'))
 commit('test: validate mature backbone initialization, caches, finite AMP and exact Base224 reuse')
 print(json.dumps(budgets),flush=True)


def bench(label):
 global ACTIVE_RECIPE
 import torch
 from torch.nn import functional as F
 phase,index=map(int,label.split(':'));ACTIVE_RECIPE='B';bind();cfg=verify()
 model=model_for(45+index%3,'convnext_tiny_B');opt=engine.optimizer(model);scaler=torch.amp.GradScaler('cuda',init_scale=128.)
 data=loader(cfg['indices']['train'],True,45);it=iter(data);model.train()
 def step():
  nonlocal it
  t=time.perf_counter()
  try:x,y,_,_=next(it)
  except StopIteration:it=iter(data);x,y,_,_=next(it)
  wait=time.perf_counter()-t;x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(x),y)
  scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
  scaler.step(opt);scaler.update();torch.cuda.synchronize();return len(y),wait
 for _ in range(4):step()
 folder=STUDY/'profile';SAVE(folder/f'ready_{phase}_{index}.json',dict(pid=os.getpid()))
 gate=folder/f'gate_{phase}.json';deadline=time.monotonic()+180
 while not gate.exists():assert time.monotonic()<deadline;time.sleep(.05)
 start=core.read_json(gate)['start_unix']
 while time.time()<start:time.sleep(.01)
 n=0;wait=0.;tick=time.perf_counter()
 while time.time()<start+12:
  amount,w=step();n+=amount;wait+=w
 elapsed=time.perf_counter()-tick
 SAVE(folder/f'bench_{phase}_{index}.json',dict(samples=n,seconds=elapsed,samples_per_second=n/elapsed,loader_wait_seconds=wait,
  peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20))


def profile():
 verify();folder=STUDY/'profile';folder.mkdir(exist_ok=True);results=[];tested=[1,2,4]
 for slots in tested:
  active=[]
  for index in range(slots):
   log=(folder/f'bench_{slots}_{index}.log').open('w')
   p=subprocess.Popen([PY,__file__,'bench','--job',f'{slots}:{index}'],cwd=ROOT,env=engine.thread_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   active.append((p,log,index))
  deadline=time.monotonic()+180
  while not all((folder/f'ready_{slots}_{i}.json').exists() for _,_,i in active):
   assert all(p.poll() is None for p,_,_ in active),'benchmark initialization failure; inspect logs'
   assert time.monotonic()<deadline;time.sleep(.2)
  SAVE(folder/f'gate_{slots}.json',dict(start_unix=time.time()+.5));stats=[]
  while any(p.poll() is None for p,_,_ in active):
   x=engine.snapshot();stats.append(x);engine.append_resource(folder/'resources.csv',dict(**x,slots=slots));time.sleep(2)
  for p,l,i in active:l.close();assert p.returncode==0
  measures=[core.read_json(folder/f'bench_{slots}_{i}.json') for _,_,i in active]
  result=dict(slots=slots,aggregate_samples_per_second=sum(x['samples_per_second'] for x in measures),workers=measures,
   mean_gpu_util=sum(x['gpu_util'] for x in stats)/len(stats),max_gpu_used_mib=max(x['gpu_used_mib'] for x in stats))
  results.append(result);SAVE(folder/'measurements.json',results)
  if slots>=4 and result['mean_gpu_util']<80 and result['aggregate_samples_per_second']>results[-2]['aggregate_samples_per_second']*1.03:
   free=engine.snapshot()['gpu_free_mib'];foot=max(x['peak_gpu_mib'] for x in measures)*1.25+700
   if free>6144+foot and (slots+2)*foot<42500 and slots+2<=os.cpu_count()//2:tested.append(slots+2)
 best=max(x['aggregate_samples_per_second'] for x in results)
 selected=next(x['slots'] for x in results if x['aggregate_samples_per_second']>=.97*best)
 SAVE(folder/'selected.json',dict(initial_slots=selected,measured_slots=[x['slots'] for x in results],rule='smallest within97percent of best total throughput'))
 commit('perf: measure real high-resolution single and concurrent baseline qualification throughput')


def run():
 verify();bind();assert core.read_json(STUDY/'preflight.json')['passed']
 with (STUDY/'pipeline.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  jobs=core.read_json(STUDY/'jobs.json');pending=[j for j in jobs if not done(j)];completed={j['id'] for j in jobs if j not in pending}
  footprints={x['arm']:max(2048,x['peak_gpu_mib']*1.25+700) for x in core.read_json(STUDY/'preflight.json')['budgets']}
  capacity=engine.snapshot()['gpu_free_mib']-6144;ceiling=min(len(jobs),os.cpu_count()//2,max(1,int(capacity/min(footprints.values()))))
  slots=min(ceiling,core.read_json(STUDY/'profile/selected.json')['initial_slots']);running={};faults=[];recent=[];next_sample=0
  logs=STUDY/'logs';logs.mkdir(exist_ok=True)
  SAVE(STUDY/'pipeline_pid.json',dict(pid=os.getpid(),started_at=core.timestamp(),identity=STUDY.name))
  while pending or running:
   changes=[]
   for pid,(p,j,l) in list(running.items()):
    if p.poll() is None:continue
    l.close();del running[pid]
    if p.returncode or not done(j):faults.append(dict(job=j['id'],returncode=p.returncode,log=str(logs/(j['id']+'.log'))))
    else:completed.add(j['id']);changes.append(j['id'])
   if time.monotonic()>=next_sample:
    stats=engine.snapshot();recent.append(stats);recent=recent[-15:]
    engine.append_resource(STUDY/'resources.csv',dict(**stats,running=len(running),pending=len(pending),slots=slots))
    if len(recent)==15 and pending and len(running)>=slots and sum(x['gpu_util'] for x in recent)/15<80 and stats['cpu_percent']<75:
     slots=min(slots+1,ceiling)
    next_sample=time.monotonic()+2
   while pending and len(running)<slots:
    # Pick the first scientifically valid queued job that fits reserved memory.
    reserved=sum(footprints[j['arm']] for _,j,_ in running.values())
    pos=next((i for i,j in enumerate(pending) if reserved+footprints[j['arm']]<=capacity and stats['gpu_free_mib']>6144+footprints[j['arm']]),None)
    if pos is None or stats['cpu_percent']>92:break
    j=pending.pop(pos);log=(logs/(j['id']+'.log')).open('a',buffering=1)
    p=subprocess.Popen([PY,__file__,'worker','--job',j['id']],cwd=ROOT,env=engine.thread_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    running[p.pid]=(p,j,log);print('launch',j['id'],p.pid,flush=True)
   if changes:commit('results: preserve mature CLE qualification '+','.join(changes))
   SAVE(STUDY/'status.json',dict(phase='training' if pending or running else 'needs_independent_audit',completed_new=len(completed),total_new=21,reused=3,
    pending=len(pending),running=[dict(job=j['id'],pid=p.pid) for p,j,l in running.values()],failures=faults,slots=slots,resources=stats,updated_at=core.timestamp()))
   if pending or running:time.sleep(.5)
  assert not faults,'Qualification fault: inspect exact logs and same-pipeline live processes; never blindly overwrite completed training'
  subprocess.run([PY,str(STUDY/'audit_qualification.py')],cwd=ROOT,env=engine.thread_env(),check=True)
  SAVE(STUDY/'status.json',dict(phase='qualification_audited_continue_structure_research',completed_new=21,reused=3,
   test_used=False,updated_at=core.timestamp(),next='Freeze selected competitive baseline and implement evidence-based CLE architecture with retrained controls'))
  commit('results: independently audit all mature backbone qualification and freeze eligible baseline')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['assets','prepare','preflight','profile','bench','worker','run']);p.add_argument('--job');a=p.parse_args()
 if a.command=='assets':assets()
 elif a.command=='prepare':prepare()
 elif a.command=='preflight':preflight()
 elif a.command=='profile':profile()
 elif a.command=='bench':bench(a.job)
 elif a.command=='worker':worker(a.job)
 else:run()
