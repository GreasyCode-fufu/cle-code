"""Lightweight own-study telemetry, including blocking inference/bootstrap intervals."""
from pathlib import Path
import csv,datetime,hashlib,json,os,subprocess,time
S=Path(__file__).resolve().parent;R=S.parents[1]
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def process(pid,expected=None,parent=None):
    root=Path('/proc')/str(pid)
    try:
        cmd=root.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if expected is not None and expected not in cmd:return None
        stat=root.joinpath('stat').read_text().rsplit(')',1)[1].split()
        if parent is not None and int(stat[1])!=parent:return None
        status=dict(line.split(':',1) for line in root.joinpath('status').read_text().splitlines() if ':' in line)
        if int(status['Uid'].split()[0])!=1000:return None
        io={k:int(v) for k,v in (line.split(':',1) for line in root.joinpath('io').read_text().splitlines())}
        return dict(pid=pid,ppid=int(stat[1]),cpu_ticks=int(stat[11])+int(stat[12]),start_ticks=int(stat[19]),rss_kib=int(status['VmRSS'].split()[0]) if 'VmRSS' in status else 0,io=io)
    except (FileNotFoundError,ProcessLookupError,PermissionError,KeyError):return None
def main():
    assert os.getuid()==1000 and S.stat().st_uid==1000
    identity=S/'runtime_sampler_identity_v2.json'
    if identity.exists():
        old=read(identity);assert process(old['pid'],'continuous_runtime_sampler.py') is None,'same own sampler already live; do not duplicate'
    save(identity,dict(pid=os.getpid(),source_sha256=sha(__file__),started_at=stamp(),training_config_sha256=sha(S/'config.json'),scope='read-only host counters and verified own manager/worker children only; no GPU context or neural work'))
    previous=None;prev_process={};previous_time=None;missing_since=None;rows=0;reason=None
    target=S/'continuous_runtime_resources.csv';details=S/'continuous_runtime_process_io.jsonl'
    while not (S/'status.json').exists():
        time.sleep(.2)
    while True:
        status=read(S/'status.json');manager=read(S/'manager_entry_identity_v2.json');manager_alive=process(manager['pid'],'run_geometry_manager.py') is not None
        own=[]
        for job in status.get('running',[]):
            p=process(job['pid'],job['job'])
            if p:
                p['job']=job['job'];own.append(p)
                try:children=(Path('/proc')/str(job['pid'])/'task'/str(job['pid'])/'children').read_text().split()
                except FileNotFoundError:children=[]
                for child in children:
                    c=process(int(child),parent=job['pid'])
                    if c:c['job']=job['job'];c['role']='DataLoader/own child';own.append(c)
        cpu={line.split()[0]:list(map(int,line.split()[1:9])) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('cpu')};percent={};iowait=0.
        if previous:
            for key,v in cpu.items():
                d=[b-a for a,b in zip(previous[key],v)];total=sum(d);percent[key]=100*(total-d[3]-d[4])/total if total else 0.
                if key=='cpu':iowait=100*d[4]/total if total else 0.
        previous=cpu;now=time.monotonic()
        for p in own:
            old=prev_process.get((p['pid'],p['start_ticks']))
            p['cpu_one_core_percent']=100*(p['cpu_ticks']-old)/(os.sysconf('SC_CLK_TCK')*(now-previous_time)) if old is not None and previous_time is not None else None
        prev_process={(p['pid'],p['start_ticks']):p['cpu_ticks'] for p in own};previous_time=now
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',');memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
        row=dict(timestamp=stamp(),unix=time.time(),gpu_util=float(gpu[0]),gpu_used_mib=float(gpu[1]),gpu_free_mib=float(gpu[2]),cpu_percent=percent.get('cpu',0),cpu_max_core_percent=max((v for k,v in percent.items() if k!='cpu'),default=0),cpu_iowait_percent=iowait,ram_used_percent=100*(1-memory['MemAvailable']/memory['MemTotal']),own_workers=len(status.get('running',[])),manager_live=manager_alive,completed=status.get('completed_new',0))
        header=not target.exists()
        with target.open('a',newline='') as file:
            w=csv.DictWriter(file,fieldnames=list(row));
            if header:w.writeheader()
            w.writerow(row)
        disk=[]
        for line in Path('/proc/diskstats').read_text().splitlines():
            v=line.split()
            if v[2].startswith(('nvme','sd','vd','dm-')):disk.append(dict(device=v[2],reads=int(v[3]),read_sectors=int(v[5]),writes=int(v[7]),written_sectors=int(v[9]),io_in_progress=int(v[11]),io_ms=int(v[12]),weighted_io_ms=int(v[13])))
        with details.open('a') as file:file.write(json.dumps(dict(**row,cpu_per_core=percent,own_processes=own,disks=disk))+'\n')
        rows+=1
        final=S/'independent_audit.json'
        if final.exists() and read(final)['passed']:reason='full module round independently audited';break
        if not manager_alive and not own:
            if missing_since is None:missing_since=now
            elif now-missing_since>30:reason='manager and verified own workers missing without final audit; inspect pipeline, no restart performed';break
        else:missing_since=None
        time.sleep(2)
    save(S/'runtime_sampler_complete_v2.json',dict(pid=os.getpid(),ended_at=stamp(),rows=rows,reason=reason,resources_sha256=sha(target),process_io_sha256=sha(details)))
if __name__=='__main__':main()
