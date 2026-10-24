# 本窗口局部几何轮研究账本

CPU纹理诊断15文件包及跨尺度完整315文件包已独立SHA交付；跨尺度全12结果不支持保留该全局统计读出。原M方向矩工作结构保留，新局部几何的预注册1321884、正式CPU bafb180、CUDA4f71144先完成，再测并发并实际开训。全部源和配置保持冻结；后处理失败先补推理/审计，不重训有效完成模型。

{
  "created_at": "2026-10-25T02:54:31.665703+08:00",
  "source_config_sha256": "3432da211c08f03d5984dd3d86bb4c9acc874e1171032a7d63270dd75745f682",
  "CPU_audit_sha256": "f2d8865a64da6f0701876bb437578426ad0357117e4e049330510216768fa834",
  "CUDA_audit_sha256": "aec05277f0db1b07de85e70fdb0838d102c721a7db97e60256655f68ebe22c96",
  "comparison_preregistration_sha256": "a755b05e445480f9f303d2f41ab4eaafa1fde54ab7d2994eba2c53a526070362",
  "jobs": 12,
  "completed": 1,
  "live_verified_workers": [
    {
      "job": "geometry_mom_affine_s46",
      "pid": 950545,
      "epochs_at_snapshot": 18,
      "actual_train_count": 2077,
      "train_index_set_sha256": "e8fe76bbcf6d8faf8dcb6f38a148341ffe81d10ac0cdc3e4dc38e9f8ccafd691",
      "samples_per_second": 418.6925126728496,
      "loader_wait_fraction": 0.007134597811536657
    }
  ],
  "profile_policy_caps": {
    "mom_affine": 1,
    "affine_only": 4,
    "mom_permuted": 4,
    "mom_dcn": 1
  },
  "actual_same_arm_jobs_at_most": 3,
  "resource_samples": 60,
  "mean_recent_gpu_util": 89.76666666666667,
  "mean_recent_CPU_percent": 6.309448658186596,
  "mean_recent_iowait": 0.0020597324095776486,
  "real_neural_training_started": true,
  "three_seed_effectiveness_not_yet_established": true,
  "overall_research_complete": false
}

整体目标active，不能因本轮结束或阴性自停。
