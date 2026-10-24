"""Sidecar-only disk/own-process I/O telemetry; training sources remain frozen."""
from pathlib import Path
import collections,datetime,json,os,time
S=Path(__file__).resolve().parent;TAG=S.name

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def snapshot():
 devices={p.name for p in Path('/sys/block').iterdir() if not p.name.startswith(('loop','ram','dm-','zram'))}
 disks={}
 for line in Path('/proc/diskstats').read_text().splitlines():
  x=line.split()
  if x[2] not in devices:continue
  v=list(map(int,x[3:]));disks[x[2]]=dict(reads=v[0],read_sectors=v[2],writes=v[4],write_sectors=v[6],io_in_progress=v[8],busy_ms=v[9])
 processes={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if TAG not in cmd:continue
   io={k:int(v) for k,v in (line.split(':') for line in (p/'io').read_text().splitlines())}
   status=(p/'status').read_text();rss=next((int(line.split()[1]) for line in status.splitlines() if line.startswith('VmRSS:')),0)
   processes[p.name]=dict(read_bytes=io.get('read_bytes',0),write_bytes=io.get('write_bytes',0),rchar=io.get('rchar',0),wchar=io.get('wchar',0),rss_kib=rss)
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 return dict(disks=disks,processes=processes)

last=snapshot();previous=time.monotonic();started=now()
(S/'resource_observer_pid.json').write_text(json.dumps(dict(pid=os.getpid(),identity=TAG,started_at=started,sample_seconds=2),indent=2))
with (S/'io_process_resources.jsonl').open('a',buffering=1) as stream:
 while True:
  time.sleep(2);tick=time.monotonic();dt=tick-previous;current=snapshot();rates={}
  for dev,v in current['disks'].items():
   old=last['disks'].get(dev,v)
   rates[dev]=dict(read_mib_s=max(0,v['read_sectors']-old['read_sectors'])*512/dt/2**20,
    write_mib_s=max(0,v['write_sectors']-old['write_sectors'])*512/dt/2**20,
    io_busy_percent=min(100,max(0,v['busy_ms']-old['busy_ms'])/(dt*10)),io_in_progress=v['io_in_progress'])
  delta=collections.Counter()
  for pid,v in current['processes'].items():
   old=last['processes'].get(pid,{k:0 for k in v})
   for k in ['read_bytes','write_bytes','rchar','wchar']:delta[k]+=max(0,v[k]-old[k])
  row=dict(timestamp=now(),interval_seconds=dt,disk_rates=rates,own_processes=current['processes'],
   own_physical_read_mib_s=delta['read_bytes']/dt/2**20,own_physical_write_mib_s=delta['write_bytes']/dt/2**20,
   own_read_syscall_mib_s=delta['rchar']/dt/2**20,own_write_syscall_mib_s=delta['wchar']/dt/2**20,
   own_rss_sum_kib=sum(v['rss_kib'] for v in current['processes'].values()),
   note='RSS sum double-counts shared mappings; physical disk stats are system-wide. mmap cache hits need not appear as read syscalls.')
  stream.write(json.dumps(row)+'\n')
  last=current;previous=tick
  if not current['processes']:
   (S/'resource_observer_complete.json').write_text(json.dumps(dict(started_at=started,ended_at=now(),reason='No matching live qualification process remains'),indent=2))
   break
