from pathlib import Path
from collections import Counter
import hashlib,json,os,subprocess,sys,traceback
S=Path(__file__).resolve().parent;R=S.parents[1];sys.path.insert(0,str(R/'experiments/ac_texture_error_evidence_window01'));import texture_evidence as e;e.S=S
def main():
    assert os.getuid()==1000 and os.environ.get('CUDA_VISIBLE_DEVICES')=='';e.save(S/'identity.json',dict(pid=os.getpid(),source_sha256=e.sha(__file__),started_at=e.stamp()))
    rows=e.read(R/'experiments/ac_dataset_v1/dataset_manifest.json');train=[v for v in rows if v['split']=='train'];assert len(train)==2077;groups=Counter(v['group_id'] for v in train);pairs=Counter((v['group_id'],v['label']) for v in train)
    assert len(groups)==182 and len(pairs)==267;diagnostic=dict(images=2077,patients=182,patient_class_pairs=267,min_images_per_patient=min(groups.values()),max_images_per_patient=max(groups.values()),top10_image_fraction=sum(sorted(groups.values())[-10:])/2077,effective_patient_count=1/sum((n/2077)**2 for n in groups.values()),classes={str(c):dict(images=sum(v['label']==c for v in train),patient_groups=len({v['group_id'] for v in train if v['label']==c}),largest_group_images=max(n for (g,k),n in pairs.items() if k==c)) for c in range(3)},scope='train metadata only; no images, no resampling/grade/label changes')
    e.save(S/'train_metadata_diagnostic.json',diagnostic)
    if not (S/'config.json').exists():
        files=[S/n for n in ['patient_alignment.py','check_patient_alignment_cpu.py','bootstrap.py','PROTO_PROTOCOL_ZH.md']]+[p for p in (S/'upstream').rglob('*') if p.is_file()]
        e.save(S/'config.json',dict(created_at=e.stamp(),identity=S.name,sources={str(p.relative_to(S)):e.sha(p) for p in files},manifest_sha256=e.sha(R/'experiments/ac_dataset_v1/dataset_manifest.json'),metadata_sha256=e.sha(S/'train_metadata_diagnostic.json'),predecessor_geometry_audit_sha256=e.sha(R/'experiments/ac_affine_geometry_structure_window01/independent_audit.json'),predecessor_geometry_direct_sha256=e.sha(R/'experiments/ac_affine_geometry_structure_window01/direct_control_comparisons.json'),lambda_aux=.05,temperature=.1,base_temperature=.1,CLE_images=0,CLE_training=0,test_predictions=0,novelty_not_established=True));e.commit('research: preregister patient-conditioned representation loss with mature and within-patient counter-hypotheses')
    cfg=e.read(S/'config.json');assert all(e.sha(S/p)==h for p,h in cfg['sources'].items()) and e.sha(R/'experiments/ac_dataset_v1/dataset_manifest.json')==cfg['manifest_sha256']
    if not (S/'numeric_preflight_cpu.json').exists():subprocess.run(['/home/login/miniconda3/envs/sunfuxiang/bin/python',str(S/'check_patient_alignment_cpu.py')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,check=True)
    assert e.read(S/'numeric_preflight_cpu.json')['passed'];e.save(S/'complete.json',dict(passed=True,config_sha256=e.sha(S/'config.json'),proof_sha256=e.sha(S/'numeric_preflight_cpu.json'),CLE_training=0,CLE_images=0,overall_research_complete=False));e.commit('test: verify patient-masked loss NumPy officialSupCon gradients degenerate cases and public initialization')
if __name__=='__main__':
    try:main()
    except Exception as exc:e.save(S/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
