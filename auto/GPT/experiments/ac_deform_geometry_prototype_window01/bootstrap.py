from pathlib import Path
import hashlib,inspect,json,os,subprocess,sys,traceback
S=Path(__file__).resolve().parent;R=S.parents[1];sys.path.insert(0,str(R/'experiments/ac_texture_error_evidence_window01'));import texture_evidence as e;e.S=S
def main():
    assert os.getuid()==1000 and os.environ.get('CUDA_VISIBLE_DEVICES')=='';e.save(S/'identity.json',dict(pid=os.getpid(),source_sha256=e.sha(__file__),started_at=e.stamp()))
    import torch,torchvision
    from torchvision.ops import deform_conv2d
    from torchvision._internally_replaced_utils import _get_extension_path
    dependencies=[Path(inspect.getfile(deform_conv2d)),Path(_get_extension_path('_C')),R/'experiments/ac_qualification_20261024_window01/qualification_model.py',R/'models/optimize/ac_module_window01/c4_moment_stem.py',R/'experiments/ac_qualification_20261024_window01/pretrained/convnext_tiny.pth']
    if not (S/'config.json').exists():
        files=[S/n for n in ['affine_geometry.py','check_geometry_cpu.py','bootstrap.py','PROTO_PROTOCOL_ZH.md']]+[p for p in (S/'upstream').rglob('*') if p.is_file()]
        e.save(S/'config.json',dict(created_at=e.stamp(),identity=S.name,thread='01a16d4e-48fe-7e22-8c4a-c6de0f4d6f76',sources={str(p.relative_to(S)):e.sha(p) for p in files},dependencies={str(p):e.sha(p) for p in dependencies},versions=dict(torch=torch.__version__,torchvision=torchvision.__version__),evidence_config_sha256=e.sha(R/'experiments/ac_texture_error_evidence_window01/config.json'),evidence_audit_sha256=e.sha(R/'experiments/ac_texture_error_evidence_window01/independent_audit.json'),CLE_images=0,neural_CLE_training=0,test_predictions=0,novelty='not established; affine/DCN/multiscale sampling prior art explicitly recorded'))
        e.commit('research: preregister bounded affine sampling prototype and faithful shared-official geometry audit')
    cfg=e.read(S/'config.json');assert all(e.sha(S/p)==h for p,h in cfg['sources'].items()) and all(e.sha(Path(p))==h for p,h in cfg['dependencies'].items())
    if not (S/'numeric_preflight_cpu.json').exists():subprocess.run(['/home/login/miniconda3/envs/sunfuxiang/bin/python',str(S/'check_geometry_cpu.py')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,check=True)
    proof=e.read(S/'numeric_preflight_cpu.json');assert proof['passed'] and proof['CLE_images']==proof['neural_CLE_training']==proof['test_predictions']==0
    e.save(S/'complete.json',dict(passed=True,config_sha256=e.sha(S/'config.json'),numeric_proof_sha256=e.sha(S/'numeric_preflight_cpu.json'),real_CLE_training=False,overall_research_complete=False));e.commit('test: verify local affine sampling against NumPy and official DCN, shared LDConv boundary semantics and public initialization')
if __name__=='__main__':
    try:main()
    except Exception as exc:e.save(S/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),action='inspect concrete prototype logs; no CLE model restart'));raise
