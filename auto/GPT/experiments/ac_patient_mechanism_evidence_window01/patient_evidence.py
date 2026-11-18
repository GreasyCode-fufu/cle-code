"""CPU-only fixed-reference feature evidence. No network updates or test images."""
from pathlib import Path
import argparse,csv,fcntl,gc,hashlib,json,os,subprocess,sys,time
S=Path(__file__).resolve().parent;R=S.parents[1]
Q=R/'experiments/ac_qualification_20261024_window01';F=R/'experiments/ac_field_structure_window01'
sys.path[:0]=[str(Q),str(R/'experiments'),str(R)]
import run_qualification as q
core=q.core;READ=core.read_json;SAVE=core.save_json
import numpy as np
import torch
from qualification_model import QualifiedBackbone

def sha(p):return core.digest(p)
def sources():return {str(p.relative_to(R)):sha(p) for p in [S/'patient_evidence.py',S/'analyze_patient_evidence.py',S/'bootstrap_patient_evidence.py',S/'PATIENT_EVIDENCE_PROTOCOL_ZH.md',Q/'run_qualification.py',Q/'qualification_model.py',R/'experiments/ac_test_feedback_window01/audit_test_feedback_v2.py']}
def commit(message):
    path=str(S.relative_to(R))
    # Other own pipelines have independent commit helpers. Do not leave our index
    # staged across their completion windows; wait without touching their processes.
    while True:
        staged=subprocess.check_output([core.GIT,'-C',str(R),'diff','--cached','--name-only'],text=True).splitlines()
        quiet=all(p.startswith(path+'/') for p in staged)
        identity=READ(F/'queue_manager_identity.json');proc=Path('/proc')/str(identity['pid'])/'cmdline'
        live=proc.exists() and b'manage_field_queue_v2.py' in proc.read_bytes()
        if live:
            status=READ(F/'queue_status_v2.json');quiet &= bool(status.get('running'))
            for item in status.get('running',[]):
                p=Path('/proc')/str(item['pid'])/'cmdline';exists=p.exists() and item['job'].encode() in p.read_bytes()
                log=F/'runs'/item['job']/'train_log.csv';epochs=len(list(csv.DictReader(log.open()))) if log.exists() else 0
                meta=F/'runs'/item['job']/'run_meta.json'
                quiet &= exists and epochs<55 and meta.exists() and time.time()-meta.stat().st_mtime>10
            for arm in ['fov_residual','field_pool','reference','no_field','no_local_norm','no_pool']:
                ready=all((F/'runs'/f'field_{arm}_s{seed}'/'validation_complete.json').exists() for seed in [45,46,47]);audit=F/'evaluations'/arm/'group_audit.json'
                if ready and (not audit.exists() or time.time()-audit.stat().st_mtime<10):quiet=False
        if quiet:break
        print('waiting for quiet own-pipeline Git window',flush=True);time.sleep(5)
    subprocess.run([core.GIT,'-C',str(R),'add','--',path],check=True)
    if subprocess.run([core.GIT,'-C',str(R),'diff','--cached','--quiet']).returncode:subprocess.run([core.GIT,'-C',str(R),'commit','-m',message],check=True)
def verify():
    q.verify();cfg=READ(S/'config.json');assert cfg['sources']==sources()
    for p,h in cfg['locked'].items():assert sha(R/p)==h,p
    return cfg

class Data:
    def __init__(self,split,indices=None):
        from torchvision import transforms as T
        assert split in ['train','val'],'only train/val diagnostic images allowed'
        cfg=READ(S/'config.json');allowed=cfg['indices'][split];self.indices=allowed if indices is None else indices
        assert set(self.indices)<=set(allowed) and len(set(self.indices))==len(self.indices),'unauthorized index'
        self.rows={r['index']:r for r in READ(S/'manifest.json')};self.recipe=cfg['recipe']
        if self.recipe=='B':
            meta=READ(Q/'cache448_meta.json');self.pos={i:p for p,i in enumerate(meta['indices'])};self.arr=np.load(Q/'cache448/images.npy',mmap_mode='r')
        else:self.arr=np.load(R/'.cache_data_study/resized_uint8.npy',mmap_mode='r');self.pos=None
        self.transform=T.Compose([T.CenterCrop(q.RECIPES[self.recipe]['crop']),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.indices)
    def __getitem__(self,pos):
        from PIL import Image
        i=self.indices[pos];r=self.rows[i];return self.transform(Image.fromarray(self.arr[i if self.pos is None else self.pos[i]])),r['label'],i

def prepare():
    if (S/'config.json').exists():return verify()
    assert os.getuid()==1000 and S.stat().st_uid==1000
    q.verify();w=READ(F/'working_reference.json');records=READ(R/'experiments/ac_dataset_v1/dataset_manifest.json');cfg=READ(Q/'config.json')
    selected=[r for r in records if r['split'] in ['train','val']];assert len(selected)==2504
    SAVE(S/'manifest.json',selected);SAVE(S/'reference.json',w)
    units=[]
    for unit in w['registration']['units']:
        comp=READ(R/unit['complete']);assert comp['epochs_completed']==60 and sha(R/unit['checkpoint'])==unit['checkpoint_sha256']==comp['checkpoint_sha256']
        units.append(unit)
    locked=[S/'manifest.json',S/'reference.json',Q/'config.json',Q/'pretrained_manifest.json',Q/'cache448_meta.json',R/'.cache_data_study/meta.json',
        R/'experiments/ac_dataset_v1/dataset_manifest.json',R/'experiments/ac_dataset_v1/dataset_config.json']
    SAVE(S/'config.json',dict(created_at=core.timestamp(),sources=sources(),locked={str(p.relative_to(R)):sha(p) for p in locked},
        seeds=[45,46,47],reference=w['arm'],family=w['family'],recipe=w['recipe'],units=units,
        indices={split:cfg['indices'][split] for split in ['train','val']},forbidden_test_example=cfg['indices']['test'][0],
        device='CPU float32; CUDA_VISIBLE_DEVICES empty',batch_size=16,workers=2,thread_candidates=[2,4,6],
        thread_selection='smallest within97percent best observed CPU samples/sec; own process nice10',
        inference_transform='deterministic center crop for BOTH train and val; no new network fitting',
        analysis=['within-class patient/within-patient feature variance','image-mean vs patient-class-mean prototype distance diagnostics on val','patient sample-weight concentration'],
        prototype_rules='raw feature arithmetic means, squared Euclidean distance, temperature1, no search; fit means only train; do not merge mixed-label patient groups',
        test_images=0,network_updates=0,independent_validation=False,
        limitation='Reference W0 was selected using earlier test feedback; encoder already fitted on all train. This is not OOF or independent verification.'))
    commit('research: freeze CPU train-val patient-heterogeneity evidence and fixed mature prototype diagnostics')
    return verify()

def preflight():
    from PIL import Image
    from torchvision import transforms as T
    from analyze_patient_evidence import prototypes
    cfg=verify();checks=[];records={r['index']:r for r in READ(S/'manifest.json')}
    for split in ['train','val']:
        ds=Data(split);size=q.RECIPES[cfg['recipe']]['resize'];crop=q.RECIPES[cfg['recipe']]['crop']
        for p in range(4):
            x,y,i=ds[p];r=records[i];assert sha(R/r['path'])==r['sha256']
            with Image.open(R/r['path']) as im:raw=im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR)
            z=T.Compose([T.CenterCrop(crop),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])(raw)
            assert torch.equal(x,z) and y==r['label'];checks.append(dict(split=split,index=i,raw_resize_crop_exact=True))
    for split,indices in [('test',None),('train',[cfg['forbidden_test_example']])]:
        try:Data(split,indices)
        except AssertionError:pass
        else:raise AssertionError('diagnostic loader accepted test')
    # Mixed-label groups are deliberately included: class-conditioned means, never one label per patient.
    X=np.arange(24,dtype=float).reshape(8,3)/7;y=np.array([0,1,0,1,2,2,1,0]);g=np.array(['a','a','a','b','b','c','c','d'])
    a,b=prototypes(X,y,g);oracle=[]
    for label in range(3):
        subset=y==label;weights=np.array([1/np.sum(subset&(g==group)) for group in g[subset]])
        oracle.append(np.average(X[subset],axis=0,weights=weights))
    assert np.allclose(b,np.array(oracle),atol=1e-12) and np.isfinite(a).all()
    SAVE(S/'preflight.json',dict(passed=True,raw_image_checks=checks,test_rejected=True,mixed_label_prototype_weighting_oracle=True,
        device='CPU',network_updates=0,test_images=0))
    commit('test: validate train-val-only feature loader and mixed-label patient-conditioned prototype arithmetic')

def state_sha(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def extract():
    cfg=verify();assert READ(S/'preflight.json')['passed'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
    try:os.nice(10)
    except PermissionError:pass
    torch.set_num_interop_threads(1);torch.set_num_threads(2)
    dataset=[Data('train'),Data('val')];allindices=cfg['indices']['train']+cfg['indices']['val'];profile=None
    for unit in cfg['units']:
        dest=S/'features'/f's{unit["seed"]}';done=S/f'features_s{unit["seed"]}.json'
        if done.exists():
            proof=READ(done);assert proof['checkpoint_sha256']==unit['checkpoint_sha256'] and all(sha(S/p)==h for p,h in proof['files'].items());continue
        assert not dest.exists(),'partial extraction exists; inspect precise log before replacing feature intermediates'
        dest.mkdir(parents=True);torch.manual_seed(unit['seed']);np.random.seed(unit['seed'])
        assert sha(R/unit['checkpoint'])==unit['checkpoint_sha256']
        asset=READ(Q/'pretrained_manifest.json')['assets'][cfg['family']];assert sha(R/asset['path'])==asset['sha256']
        model=QualifiedBackbone(cfg['family'],Q/'pretrained').eval();weights=torch.load(R/unit['checkpoint'],map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in weights.values() if v.is_floating_point());model.load_state_dict(weights,strict=True);del weights
        before=state_sha(model)
        if profile is None:
            x=torch.stack([dataset[0][p][0] for p in range(4)]);measures=[]
            with torch.inference_mode():
                for threads in cfg['thread_candidates']:
                    torch.set_num_threads(threads);model(x);start=time.perf_counter()
                    for _ in range(3):model(x)
                    seconds=time.perf_counter()-start;measures.append(dict(threads=threads,samples=12,seconds=seconds,samples_per_second=12/seconds))
            best=max(m['samples_per_second'] for m in measures);threads=next(m['threads'] for m in measures if m['samples_per_second']>=.97*best)
            profile=dict(measurements=measures,threads=threads,device='CPU',batch_for_resource_check=4);SAVE(S/'cpu_profile.json',profile)
        torch.set_num_threads(profile['threads']);start=time.perf_counter();features=[];logits=[];seen=[];labels=[]
        for split,data in zip(['train','val'],dataset):
            dl=torch.utils.data.DataLoader(data,batch_size=16,shuffle=False,num_workers=2,generator=torch.Generator().manual_seed(unit['seed']),persistent_workers=False)
            with torch.inference_mode():
                for x,y,index in dl:
                    tick=time.perf_counter();z=model.encoder(x);out=model.head(z);assert torch.isfinite(z).all() and torch.isfinite(out).all()
                    features.append(z.numpy());logits.append(out.numpy());seen.extend(index.tolist());labels.extend(y.tolist())
                    if len(seen)%256<16:print('CPU features',unit['seed'],split,len(seen),len(allindices),'samples/sec',len(seen)/(time.perf_counter()-start),flush=True)
                    q.engine.append_resource(S/'cpu_resources.csv',dict(**q.engine.snapshot(),seed=unit['seed'],split=split,images_done=len(seen),cpu_threads=profile['threads'],batch_seconds=time.perf_counter()-tick))
        assert seen==allindices and before==state_sha(model)
        arrays=dict(features=np.concatenate(features),logits=np.concatenate(logits),indices=np.array(seen),labels=np.array(labels))
        for name,array in arrays.items():np.save(dest/f'{name}.npy',array)
        SAVE(done,dict(seed=unit['seed'],checkpoint_sha256=unit['checkpoint_sha256'],state_sha256=before,state_unchanged=True,
            config_sha256=sha(S/'config.json'),source_hashes=sources(),images=2504,train=2077,val=427,test=0,network_updates=0,
            shape=list(arrays['features'].shape),seconds=time.perf_counter()-start,threads=profile['threads'],device='CPU float32',
            files={str(p.relative_to(S)):sha(p) for p in dest.glob('*.npy')}))
        del model,features,logits,arrays;gc.collect();commit('evidence: preserve fixed-reference CPU features and immutable state for seed'+str(unit['seed']))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','preflight','extract']);a=parser.parse_args()
    lock=(S/'evidence.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[a.command]()
