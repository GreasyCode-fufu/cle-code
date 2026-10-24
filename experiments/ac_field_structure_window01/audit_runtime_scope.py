"""Separate measured throughput windows from process teardown in saved telemetry."""
from pathlib import Path
import csv,datetime,json,sys
import numpy as np
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import run_field_structure as f

def main():
    f.verify();profiles={}
    for folder in [S/'profile',*sorted((S/'runtime_profiles').glob('*'))]:
        if not (folder/'measurements.json').exists():continue
        entries=f.READ(folder/'measurements.json');resources=list(csv.DictReader((folder/'resources.csv').open()));result=[]
        for entry in entries:
            slots=entry['slots'];gate=f.READ(folder/f'gate_{slots}.json')['start_unix']
            end=gate+min(worker['seconds'] for worker in entry['workers'])
            same=[r for r in resources if int(r['slots'])==slots]
            active=[r for r in same if gate+1 <= datetime.datetime.fromisoformat(r['timestamp']).timestamp() <= end-1]
            result.append(dict(slots=slots,aggregate_samples_per_second=entry['aggregate_samples_per_second'],
                originally_reported_entire_process_gpu_mean=entry['mean_gpu_util'],entire_process_samples=len(same),
                guarded_active_window_gpu_mean=float(np.mean([float(r['gpu_util']) for r in active])) if active else None,
                guarded_active_samples=len(active),active_window_unix=[gate+1,end-1],
                throughput_measurement_unchanged=True))
        profiles[folder.name]=dict(source_measurement_sha256=f.sha(folder/'measurements.json'),phases=result)
    recent=list(csv.DictReader((S/'continuous_resources_v2.csv').open()))[-30:]
    report=dict(created_at=f.core.timestamp(),profiles=profiles,recent_continuous_samples=len(recent),
        recent_gpu_mean=float(np.mean([float(r['gpu_util']) for r in recent])),
        recent_cpu_mean=float(np.mean([float(r['cpu_percent']) for r in recent])),
        recent_iowait_mean=float(np.mean([float(r['cpu_iowait_percent']) for r in recent])),
        note='Original GPU averages poll until benchmark processes exit and can include DataLoader/interpreter teardown. Samples/sec uses timed step loop. Guarded window excludes first/last1s; continuous training utilization is separate. No benchmark/model/result rerun.')
    f.SAVE(S/'runtime_scope_audit.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
