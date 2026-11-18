"""Resource-aware finite job queue shared by new AC experiments."""
import collections
import csv
import fcntl
import os
import subprocess
import sys
import time
import numpy as np


def run_queue(study):
    core = study.core
    folder = study.STUDY
    with (folder / 'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = study.verify()
        assert core.read_json(folder / 'preflight_validation.json')['passed']
        jobs = core.read_json(folder / 'final_jobs.json')
        pending = [j for j in jobs if not study.done(j)]
        running = {}
        retries = collections.Counter()
        logs = folder / 'logs'
        logs.mkdir(exist_ok=True)
        slots = cfg['initial_parallel']
        start = time.monotonic()
        next_sample = 0
        recent = []
        while pending or running:
            completed = []
            for pid, (process, job, log) in list(running.items()):
                if process.poll() is None:
                    continue
                log.close()
                del running[pid]
                if process.returncode or not study.done(job):
                    retries[job['id']] += 1
                    if retries[job['id']] <= 1:
                        pending.append(job)
                    else:
                        raise RuntimeError(job['id'] + ' failed twice; inspect logs and live children')
                else:
                    completed.append(job['id'])
            if time.monotonic() >= next_sample:
                stats = study.snapshot()
                stats.update(running=len(running), pending=len(pending), slots=slots)
                recent.append(stats)
                recent = recent[-15:]
                path = folder / 'resource_samples.csv'
                exists = path.exists()
                with path.open('a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(stats))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(stats)
                if time.monotonic() - start > 30 and len(recent) >= 10 and pending and len(running) >= slots:
                    if (np.mean([x['gpu_util'] for x in recent]) < 80 and np.mean([x['cpu_percent'] for x in recent]) < 75
                            and stats['gpu_free_mib'] > cfg['reserve_gpu_mib'] + cfg['new_job_gpu_mib']):
                        slots = min(slots + 1, cfg['max_parallel'])
                next_sample = time.monotonic() + 2
            while pending and len(running) < slots:
                if study.resources()['gpu_free_mib'] < cfg['reserve_gpu_mib'] + cfg['new_job_gpu_mib'] or (running and stats['cpu_percent'] > 90):
                    break
                job = pending.pop(0)
                log = (logs / (job['id'] + '.log')).open('a', buffering=1)
                env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
                process = subprocess.Popen([sys.executable, study.__file__, 'worker', '--job', job['id']], cwd=study.ROOT,
                    env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                running[process.pid] = (process, job, log)
                print('[launch]', job['id'], process.pid, flush=True)
            if completed:
                study.commit('results: complete AC backbone/recipe jobs ' + ','.join(completed))
            core.save_json(folder / 'status.json', dict(phase='training', total=len(jobs), completed=len(jobs)-len(pending)-len(running),
                pending=len(pending), slots=slots, resources=stats, running=[dict(job=j['id'], pid=p.pid) for p,j,_ in running.values()], updated_at=core.timestamp()))
            if pending or running:
                time.sleep(.5)
        core.save_json(folder / 'results.json', [core.read_json(folder/'runs'/j['id']/'validation_complete.json') for j in jobs])
        study.commit('results: complete AC backbone/recipe study for independent audit')
        core.save_json(folder/'status.json', dict(phase='complete_needs_independent_audit', completed=len(jobs), total=len(jobs), updated_at=core.timestamp()))
