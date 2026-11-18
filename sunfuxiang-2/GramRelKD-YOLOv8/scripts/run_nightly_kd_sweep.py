#!/usr/bin/env python3
"""Resumable, validation-ranked nightly composite-KD sweep (single GPU)."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/cle/sunfuxiang-2/ultralytics-main")
PYTHON = "/home/login/miniconda3/envs/yolo/bin/python"
ANALYSIS = ROOT / "runs/analysis/nightly_kd_sweep_20260801"
TEACHER = ROOT / "runs/detect/runs/detect/hongkong_L1-96/weights/best.pt"
EXPECTED_INIT = "b5c987921cb2eae9d9536c8c632c619145e7d56a992f37d9badff8a368ae25b6"
MAX_HOURS = 8.75

EXPERIMENTS = [
    {"name":"E01_GramMTRD_75_25","components":["gram","mtrd"],"alloc":{"gram":.75,"mtrd":.25},"fixed":{"gram":.0323743788,"mtrd":.0256157314}},
    {"name":"E02_GramSoftFG_70_30","components":["gram","foreground"],"alloc":{"gram":.70,"foreground":.30}},
    {"name":"E03_GramAT_80_20","components":["gram","attention"],"alloc":{"gram":.80,"attention":.20}},
    {"name":"E04_GramAdvP5_80_20","components":["gram","adv_p5"],"alloc":{"gram":.80,"adv_p5":.20},"gan":True},
    {"name":"E05_GramDecoupledCB_60_25_15","components":["gram","foreground_cb","background"],"alloc":{"gram":.60,"foreground_cb":.25,"background":.15}},
    {"name":"E06_GramResponseKD_70_30","components":["gram","response"],"alloc":{"gram":.70,"response":.30}},
    {"name":"E07_GramAdvP4P5_80_20","components":["gram","adv_p4","adv_p5"],"alloc":{"gram":.80,"adv_p4":.10,"adv_p5":.10},"gan":True},
]
FALLBACKS = [
    {"name":"F01_GramKD_2p5Percent","components":["gram"],"alloc":{"gram":.5},"fixed":{"gram":.0215829192},"fallback":True},
    {"name":"F02_GramKD_10Percent","components":["gram"],"alloc":{"gram":2.0},"fixed":{"gram":.0863316768},"fallback":True},
    {"name":"F03_GramKD_P3P4","components":["gram"],"alloc":{"gram":1.0},"layers":"1,1,0","fallback":True},
]
PROTECTED = ["runs/detect/hongkong_SR1_P4Only_RelKD","runs/detect/hongkong_SR1_P4Only_PairedNoKD","runs/detect/hongkong_SR1_P4Only_MTRD",
             "runs/analysis/kd_formal_result","runs/analysis/kd_paired_nokd_result","runs/analysis/mtrd_formal_result"]


def now(): return datetime.now(timezone.utc).astimezone().isoformat()
def write_json(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
def read_json(path, default):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return default
def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def tree_hash(path):
    h=hashlib.sha256()
    if not path.exists(): return None
    for p in sorted(x for x in path.rglob("*") if x.is_file()): h.update(str(p.relative_to(path)).encode()); h.update(file_hash(p).encode())
    return h.hexdigest()
def core_hash():
    h=hashlib.sha256()
    for p in sorted((ROOT/"ultralytics").rglob("*.py")): h.update(str(p.relative_to(ROOT)).encode()); h.update(file_hash(p).encode())
    return h.hexdigest()
def log(message):
    line=f"[{now()}] {message}"; print(line,flush=True)
    with (ANALYSIS/"nightly_console.log").open("a",encoding="utf-8") as f: f.write(line+"\n")
def run(command, log_path, attempts=1):
    for attempt in range(1, attempts+1):
        log(f"RUN attempt={attempt}: {' '.join(map(str,command))}")
        with log_path.open("a",encoding="utf-8") as f:
            f.write(f"\n=== {now()} attempt {attempt} ===\n"); result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode==0: return 0
        log(f"command failed rc={result.returncode}; {'retrying once after cleanup' if attempt<attempts else 'no retries left'}")
        if attempt<attempts:
            subprocess.run([PYTHON,"-c","import gc,torch;gc.collect();torch.cuda.empty_cache()"],cwd=ROOT); time.sleep(5)
    return result.returncode
def gpu_busy():
    q=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],capture_output=True,text=True)
    return bool(q.stdout.strip())
def wait_for_gpu(deadline):
    while gpu_busy():
        if time.time()+600>=deadline: return False
        log("GPU has another compute process; waiting 60 seconds"); time.sleep(60)
    return True
def best_epoch(results_csv):
    rows=list(csv.DictReader(open(results_csv,encoding="utf-8"))); key="metrics/mAP50-95(B)"
    if not rows: return None
    row=max(rows,key=lambda x:float(x.get(key,0) or 0)); return int(float(row["epoch"]))+1


def calibration(exp, path):
    if exp.get("fixed"):
        data={"status":"SUCCESS","experiment":exp["name"],"source":"fixed verified Gram/MTRD five-percent anchors",
              "allocations":exp["alloc"],"final_component_weights":exp["fixed"],"measurement_batches":20 if exp.get("fallback") else 0}
        write_json(path,data); return data
    command=[PYTHON,"scripts/calibrate_composite_kd.py","--experiment",exp["name"],"--components",",".join(exp["components"]),
             "--allocations",json.dumps(exp["alloc"]),"--output",str(path),"--layer-weights",exp.get("layers","1,1,1")]
    if run(command,ANALYSIS/f"{exp['name']}_calibration.log",attempts=2): raise RuntimeError("calibration failed twice")
    return read_json(path,{})


def preflight(exp, core_before, teacher_before, unit_ok):
    result={"experiment":exp["name"],"started":now(),"unit_tests":unit_ok,"attempts":[]}; cal_path=ANALYSIS/f"{exp['name']}_calibration.json"
    try:
        cal=calibration(exp,cal_path); weights=cal["final_component_weights"]; weights_path=ANALYSIS/f"{exp['name']}_weights.json"; write_json(weights_path,weights)
        class_path=None
        if "foreground_cb" in exp["components"]:
            class_path=ANALYSIS/f"{exp['name']}_class_weights.json"; write_json(class_path,cal["class_weights"])
        for attempt in (1,2):
            inspect_json=ANALYSIS/f"{exp['name']}_inspect.json"
            base=[PYTHON,"scripts/composite_distill_train.py","--experiment",exp["name"],"--weights-json",str(weights_path),"--layer-weights",exp.get("layers","1,1,1")]
            if class_path: base += ["--class-weights-json",str(class_path)]
            rc1=run(base+["--mode","inspect","--result-json",str(inspect_json)],ANALYSIS/f"{exp['name']}_preflight.log")
            smoke_json=ANALYSIS/f"{exp['name']}_smoke.json"
            rc2=run(base+["--mode","smoke","--result-json",str(smoke_json)],ANALYSIS/f"{exp['name']}_preflight.log") if rc1==0 else 99
            inspect=read_json(inspect_json,{}); smoke=read_json(smoke_json,{})
            checks={"inspect_ok":rc1==0,"smoke_ok":rc2==0,"finite":inspect.get("finite"),"student_changed":inspect.get("student_changed"),
                    "teacher_eval":inspect.get("teacher_eval"),"teacher_hash":inspect.get("teacher_hash_unchanged"),
                    "hooks_once":all(all(int(v)==1 for v in d.values()) for d in inspect.get("hook_counts",[])),
                    "checkpoint_standalone":inspect.get("checkpoint_standalone"),"initialization_match":inspect.get("student_initialization_hash")==EXPECTED_INIT,
                    "smoke_clean":smoke.get("checkpoint_clean"),"core_unchanged":core_hash()==core_before,"teacher_file_unchanged":file_hash(TEACHER)==teacher_before}
            result["attempts"].append(checks)
            if unit_ok and all(checks.values()):
                result.update({"status":"PASSED","ended":now(),"weights":weights,"calibration":str(cal_path)}); return result
            log(f"{exp['name']} preflight attempt {attempt} failed; applying cleanup/retry recovery")
            time.sleep(5)
        raise RuntimeError("preflight failed twice")
    except Exception as e:
        result.update({"status":"FAILED","ended":now(),"reason":repr(e)}); return result


def train_and_validate(exp, weights, class_weights_path):
    name=exp["name"]; run_dir=ROOT/"runs/detect"/name
    if run_dir.exists():
        suffix=datetime.now().strftime("%H%M%S"); name=f"{name}_{suffix}"; run_dir=ROOT/"runs/detect"/name
    weights_path=ANALYSIS/f"{exp['name']}_weights.json"; train_json=ANALYSIS/f"{exp['name']}_training.json"
    command=[PYTHON,"scripts/composite_distill_train.py","--mode","train","--experiment",exp["name"],"--weights-json",str(weights_path),
             "--name",name,"--layer-weights",exp.get("layers","1,1,1"),"--result-json",str(train_json)]
    if class_weights_path: command += ["--class-weights-json",str(class_weights_path)]
    started=time.time(); rc=run(command,ANALYSIS/f"{exp['name']}_formal.log"); elapsed=time.time()-started
    train=read_json(train_json,{})
    if rc or train.get("status")!="SUCCESS": raise RuntimeError(f"formal training failed rc={rc}")
    best=Path(train["best"]); val_path=ANALYSIS/f"{exp['name']}_validation.json"
    rc=run([PYTHON,"scripts/evaluate_nightly_model.py","--weights",str(best),"--split","val","--output",str(val_path)],ANALYSIS/f"{exp['name']}_validation.log")
    if rc: raise RuntimeError("validation failed")
    val=read_json(val_path,{}); return train,val,elapsed,best_epoch(Path(train["save_dir"])/"results.csv")


def write_table(path, rows, fields):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def write_progress_tables(state):
    done=[x for x in state["experiments"] if x.get("status")=="COMPLETED"]
    val_fields=["experiment","validation_map50_95","validation_map75","precision","recall","spalling_ap50_95","crack_ap50_95","moisture_ap50_95"]
    resource_fields=["experiment","training_seconds","peak_cuda_memory_mb","fused_params","unfused_params","gflops"]
    write_table(ANALYSIS/"validation_results.csv",done,val_fields); write_table(ANALYSIS/"resource_log.csv",done,resource_fields)


def finalize(state, core_before, teacher_before, protected_before):
    completed=[x for x in state["experiments"] if x.get("status")=="COMPLETED"]
    ranked=sorted(completed,key=lambda x:(x["validation_map50_95"],x["validation_map75"],x.get("moisture_ap50_95",0),x.get("crack_ap50_95",0)),reverse=True)
    selected=ranked[:2]; best_gan=next((x for x in ranked if x.get("gan")),None)
    if best_gan and best_gan not in selected: selected.append(best_gan)
    tests=[]
    for row in selected[:3]:
        out=ANALYSIS/f"{row['experiment']}_test.json"; rc=run([PYTHON,"scripts/evaluate_nightly_model.py","--weights",row["best_weight"],"--split","test","--output",str(out)],ANALYSIS/f"{row['experiment']}_test.log")
        if rc==0:
            value=read_json(out,{}); tests.append({"experiment":row["experiment"],**value["overall"],"spalling_ap50_95":value["classes"].get("spalling",{}).get("ap50_95"),"crack_ap50_95":value["classes"].get("crack",{}).get("ap50_95"),"moisture_ap50_95":value["classes"].get("moisture",{}).get("ap50_95"),"fused_params":value["fused_params"],"unfused_params":value["unfused_params"],"gflops":value["gflops"]})
    fields=["experiment","status","training_seconds","best_epoch","validation_map50_95","validation_map75","precision","recall","spalling_ap50_95","crack_ap50_95","moisture_ap50_95","best_weight","peak_cuda_memory_mb","initialization_hash","gan","fallback_for"]
    write_table(ANALYSIS/"full_results.csv",state["experiments"],fields)
    if tests: write_table(ANALYSIS/"test_results.csv",tests,list(tests[0]))
    integrity={"teacher_hash_before":teacher_before,"teacher_hash_after":file_hash(TEACHER),"teacher_unchanged":file_hash(TEACHER)==teacher_before,
               "core_hash_before":core_before,"core_hash_after":core_hash(),"core_unchanged":core_hash()==core_before,
               "protected_results":{p:{"before":protected_before[p],"after":tree_hash(ROOT/p),"unchanged":protected_before[p]==tree_hash(ROOT/p)} for p in PROTECTED},
               "student_inference_complexity_unchanged":all((x.get("fused_params")==2603129 and abs(x.get("gflops",0)-7.044096)<1e-6) for x in tests)}
    write_json(ANALYSIS/"source_integrity_report.json",integrity)
    best=ranked[0] if ranked else None; best_test=max(tests,key=lambda x:x["map50_95"]) if tests else None
    status="SUCCESS" if len(completed)==7 and integrity["teacher_unchanged"] and integrity["core_unchanged"] else "PARTIAL_SUCCESS" if completed else "FAILED"
    end=time.time(); state.update({"status":status,"end_time":now(),"elapsed_hours":(end-state["start_epoch"])/3600,"tests":tests}); write_json(ANALYSIS/"sweep_state.json",state)
    lines=["# Nightly KD sweep report","",f"Status: {status}",f"Completed full 300-epoch experiments: {len(completed)}.",
           f"Validation leader: {best['experiment'] if best else 'none'} ({best['validation_map50_95']:.6f})" if best else "Validation leader: none.",
           f"Best tested exploratory model: {best_test['experiment']} ({best_test['map50_95']:.6f})" if best_test else "No exploratory model reached test evaluation.",
           "", "Gram RelKD remains the fixed final main method. Every new method is an exploratory extension; single-seed differences are not statistical significance."]
    (ANALYSIS/"final_report.md").write_text("\n".join(lines)+"\n")
    (ANALYSIS/"method_comparison.md").write_text("# Method comparison\n\nFixed references: NoKD 0.42120, MTRD 0.432186, Gram RelKD 0.435680 test mAP50-95. See full_results.csv and test_results.csv.\n")
    gan_rows=[x for x in completed if x.get("gan")]; (ANALYSIS/"gan_stability_report.md").write_text("# GAN stability\n\n"+("\n".join(f"- {x['experiment']}: completed; metrics in {x['metrics_csv']}" for x in gan_rows) if gan_rows else "No GAN full run completed.")+"\n")
    (ANALYSIS/"recommended_next_action.md").write_text("# Recommended next action\n\nKeep Gram RelKD as the official main method. Treat any leader as a candidate extension requiring explicit user approval and multi-seed confirmation.\n")
    log("FINAL SUMMARY")
    summary={"Nightly sweep status":status,"Start time":state["start_time"],"End time":state["end_time"],"Elapsed hours":state["elapsed_hours"],
             "Completed 300-epoch experiments":len(completed),"Skipped experiments":len([x for x in state["experiments"] if x.get("status")=="SKIPPED"]),
             "Fallback experiments":len([x for x in completed if x.get("fallback_for")]),"Failed experiments":len([x for x in state["experiments"] if x.get("status")=="FAILED"]),
             "Best validation experiment":best["experiment"] if best else None,"Best validation mAP50-95":best["validation_map50_95"] if best else None,
             "Best GAN experiment":best_gan["experiment"] if best_gan else None,"Best GAN validation mAP50-95":best_gan["validation_map50_95"] if best_gan else None,
             "Models tested on test split":[x["experiment"] for x in tests],"Best exploratory test mAP50-95":best_test["map50_95"] if best_test else None,
             "Gram RelKD test mAP50-95":0.435680,"Teacher hash unchanged":integrity["teacher_unchanged"],"Core source unchanged":integrity["core_unchanged"],
             "Student inference complexity unchanged":integrity["student_inference_complexity_unchanged"],"Final main method":"Gram RelKD","Report directory":str(ANALYSIS)}
    write_json(ANALYSIS/"final_summary.json",summary); print(json.dumps(summary,indent=2),flush=True)


def main():
    ANALYSIS.mkdir(parents=True,exist_ok=True); state_path=ANALYSIS/"sweep_state.json"; existing=read_json(state_path,{})
    if existing.get("status")=="FAILED" and not any(x.get("status")=="COMPLETED" for x in existing.get("experiments",[])):
        archive=ANALYSIS/f"prelaunch_failure_state_{int(time.time())}.json"; write_json(archive,existing)
        log(f"Archived zero-completion prelaunch failure state to {archive.name}; starting a clean timed sweep")
        existing={}
    state=existing or {"status":"RUNNING","start_time":now(),"start_epoch":time.time(),"experiments":[],"skipped":[],"failures":[]}
    core_before=state.setdefault("core_hash_before",core_hash()); teacher_before=state.setdefault("teacher_hash_before",file_hash(TEACHER))
    protected_before=state.setdefault("protected_hashes_before",{p:tree_hash(ROOT/p) for p in PROTECTED}); deadline=state["start_epoch"]+MAX_HOURS*3600
    write_json(ANALYSIS/"experiment_manifest.json",{"max_hours":MAX_HOURS,"experiments":EXPERIMENTS,"fallbacks":FALLBACKS,"final_main_method":"Gram RelKD"})
    write_json(state_path,state); unit_log=ANALYSIS/"unit_tests.log"
    unit_ok=run([PYTHON,"-m","pytest","-q","tests/test_composite_kd.py","tests/test_adversarial_feature_kd.py","tests/test_nightly_kd_pipeline.py"],unit_log,attempts=2)==0
    fallback_index=state.get("fallback_index",0); completed_names={x["planned_experiment"] for x in state["experiments"] if x.get("status") in {"COMPLETED","FAILED","SKIPPED"}}
    e4_unstable=False
    for planned in EXPERIMENTS:
        if planned["name"] in completed_names: continue
        durations=[x["training_seconds"] for x in state["experiments"] if x.get("status")=="COMPLETED"]
        estimate=statistics.median(durations) if durations else 3600.0
        if planned.get("gan"): estimate*=1.20
        if deadline-time.time() < estimate+600:
            state["experiments"].append({"planned_experiment":planned["name"],"experiment":planned["name"],"status":"SKIPPED","reason":"insufficient time budget"}); write_json(state_path,state); continue
        if shutil.disk_usage(ROOT).free < 20*1024**3:
            state["experiments"].append({"planned_experiment":planned["name"],"experiment":planned["name"],"status":"SKIPPED","reason":"less than 20 GiB free"}); continue
        if not wait_for_gpu(deadline):
            state["experiments"].append({"planned_experiment":planned["name"],"experiment":planned["name"],"status":"SKIPPED","reason":"GPU remained busy"}); continue
        candidate=planned
        if planned["name"].startswith("E07") and e4_unstable: candidate=None
        pre=preflight(candidate,core_before,teacher_before,unit_ok) if candidate else {"status":"FAILED","reason":"E4 engineering instability"}
        if pre["status"]!="PASSED":
            state["failures"].append({"experiment":planned["name"],"stage":"preflight","detail":pre});
            if fallback_index<len(FALLBACKS): candidate=FALLBACKS[fallback_index]; fallback_index+=1; state["fallback_index"]=fallback_index; pre=preflight(candidate,core_before,teacher_before,unit_ok)
        state.setdefault("preflights",[]).append(pre); write_json(ANALYSIS/"preflight_results.json",state["preflights"]); write_json(state_path,state)
        if pre["status"]!="PASSED":
            state["experiments"].append({"planned_experiment":planned["name"],"experiment":candidate["name"],"status":"FAILED","reason":"candidate and fallback preflight failed"}); continue
        class_path=ANALYSIS/f"{candidate['name']}_class_weights.json" if "foreground_cb" in candidate["components"] else None
        try:
            train,val,elapsed,epoch=train_and_validate(candidate,pre["weights"],class_path)
            overall=val["overall"]; classes=val["classes"]
            record={"planned_experiment":planned["name"],"experiment":candidate["name"],"status":"COMPLETED","training_seconds":elapsed,"best_epoch":epoch,
                    "validation_map50_95":overall["map50_95"],"validation_map75":overall["map75"],"precision":overall["precision"],"recall":overall["recall"],
                    "spalling_ap50_95":classes.get("spalling",{}).get("ap50_95",0),"crack_ap50_95":classes.get("crack",{}).get("ap50_95",0),"moisture_ap50_95":classes.get("moisture",{}).get("ap50_95",0),
                    "best_weight":train["best"],"metrics_csv":train["metrics_csv"],"peak_cuda_memory_mb":train["peak_cuda_memory_mb"],
                    "initialization_hash":train["student_initialization_hash"],"teacher_hash_unchanged":train["teacher_hash_unchanged"],"checkpoint_clean":train["checkpoint_clean"],
                    "fused_params":val["fused_params"],"unfused_params":val["unfused_params"],"gflops":val["gflops"],"gan":bool(candidate.get("gan")),
                    "fallback_for":planned["name"] if candidate.get("fallback") else None}
            state["experiments"].append(record); e4_unstable=False
        except Exception as e:
            state["experiments"].append({"planned_experiment":planned["name"],"experiment":candidate["name"],"status":"FAILED","reason":repr(e),"gan":bool(candidate.get("gan"))})
            state["failures"].append({"experiment":candidate["name"],"stage":"formal","reason":repr(e)}); e4_unstable=planned["name"].startswith("E04")
        write_json(ANALYSIS/"failures.json",state["failures"]); write_json(ANALYSIS/"initialization_hashes.json",{x["experiment"]:x.get("initialization_hash") for x in state["experiments"]}); write_progress_tables(state); write_json(state_path,state)
    finalize(state,core_before,teacher_before,protected_before); return 0


if __name__=="__main__": raise SystemExit(main())
