"""Clarify determinant metadata before permutation; no neural training or CLE image."""
from pathlib import Path
import json,os,sys
import numpy as np
import torch
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
from models.optimize.ac_affine_geometry_window01.affine_geometry import affine_offsets
m.verify();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';rng=torch.Generator().manual_seed(20261025);raw=torch.randn(1,6,1,1,generator=rng,dtype=torch.double)*.5;permutation=torch.randperm(49,generator=torch.Generator().manual_seed(20261025));a,j=affine_offsets(raw,7,return_jacobian=True);b=affine_offsets(raw,7,permutation)
aa=a.reshape(49,2).numpy();bb=b.reshape(49,2).numpy();yy,xx=np.meshgrid(np.arange(-3,4),np.arange(-3,4),indexing='ij');grid=np.stack((yy.ravel(),xx.ravel()),1);design=np.concatenate((grid,np.ones((49,1))),1)
fit_a=np.linalg.lstsq(design,grid+aa,rcond=None)[0];fit_b=np.linalg.lstsq(design,grid+bb,rcond=None)[0];residual_a=float(np.max(np.abs(design@fit_a-grid-aa)));residual_b=float(np.max(np.abs(design@fit_b-grid-bb)));assert residual_a<1e-12 and residual_b>.01
assert np.array_equal(bb,aa[permutation.numpy()]) and abs(np.linalg.norm(aa)-np.linalg.norm(bb))<1e-12
record=dict(passed=True,source_config_sha256=m.sha(S/'config.json'),original_preflight_sha256=m.sha(S/'preflight.json'),audit_source_sha256=m.sha(__file__),field='geometry_diagnostics.local_determinant_range',scope_by_arm=dict(mom_affine='actual unpermuted local affine generator A',affine_only='actual unpermuted local affine generator A',mom_permuted='A before kernel-point permutation; this determinant does not describe the final permuted sampling grid',mom_dcn='no affine determinant field'),CPU_unpermuted_affine_fit_residual=residual_a,CPU_permuted_nonaffine_fit_residual=residual_b,permuted_offset_marginal_exact=True,no_global_topology_or_equivariance_claim=True,neural_training=0,CLE_images=0,test_images=0,model_config_and_training_unchanged=True)
m.SAVE(S/'preflight_geometry_scope_audit.json',record)
p=S/'STRUCTURE_AND_USE_ZH.md';p.write_text(p.read_text()+'\n数值元数据补充：preflight中mom_permuted的local_determinant_range指打乱前生成offset的A；打乱后的采样格不再是该局部仿射映射，不能据该数值宣称保持方向。CPU独立最小二乘拟合和offset边际精确核验见preflight_geometry_scope_audit.json；原预检/源码/config/训练均保持原样。\n')
print(json.dumps(record),flush=True)
