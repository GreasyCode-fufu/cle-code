# 本新窗口研究账本：跨尺度轮

历史S1/Q/T、患者/方向证据及成熟C4/预算对照均已完成，保持冻结、不重训。F18几何方法全审计，未获稳定增量，完整曲线/证据350文件已交付。M18共享模块方法全审计，原方向矩作为工作结构保留，其C4对B0+1.476485百分点严格未达1.5；444文件完整包已交付。P12方向矩消融/累计全审计，mean_only反向支持sigma，但ECA/aug不稳定，288文件完整包已交付。

本轮机制、数据、预算、假设与对照在MODULE_PROTOCOL_ZH.md、config.json、jobs.json及comparison_preregistration.json冻结。Git f1c8b52预注册、f53f263CPU、df8e174CUDA，88b50cb直接对照/官方许可补充，随后并发测量提交及真实启动。每个有效操作/阴性/完整权重继续入Git，后处理失败只补推理不盲重训。

{
  "created_at": "2026-10-25T02:10:08.851033+08:00",
  "source_config_sha256": "346b437341364f4cb8377dd251c7aa127b164aba25559ab713c3d283711dc69a",
  "CPU_audit_sha256": "7d77b53e630ebcda90bbbdf63149a43a02f3fdccb80806aaafcb21c03f9a5dc5",
  "CUDA_audit_sha256": "773f1263c720b736b09fce0877ae72556083ad3c98bdadc7a570bd5ee7fa5c7b",
  "comparison_preregistration_sha256": "3fef1193c6ae0ef995f18375774aa0cfc4a766f366b02ce595a8169552e98cb6",
  "jobs": 12,
  "running": [
    {
      "job": "crossscale_mom_corr_s45",
      "pid": 929166,
      "epochs_at_snapshot": 59,
      "latest_actual_train_count": 2077,
      "train_index_set_sha256": "e8fe76bbcf6d8faf8dcb6f38a148341ffe81d10ac0cdc3e4dc38e9f8ccafd691",
      "latest_train_samples_per_second": 905.6270554642724,
      "loader_wait_fraction": 0.015074631954177532
    }
  ],
  "profile": {
    "slots_by_arm": {
      "mom_corr": 1,
      "corr_only": 1,
      "mom_shuffle": 1,
      "mom_bilinear": 1
    },
    "rule": "smallest within97percent of measured best; active-kernel GPU samples, not interpreter startup;6GiB reserve"
  },
  "recent_samples": 60,
  "recent_mean_gpu_util": 91.78333333333333,
  "recent_mean_CPU_percent": 8.317463148710896,
  "recent_max_core_percent": 96.55172413793103,
  "recent_mean_iowait_percent": 0.002057967944101314,
  "full_reference_probability_reproductions": [
    {
      "kind": "plain",
      "seed": 45,
      "full427_max_probability_error": 0.0
    },
    {
      "kind": "plain",
      "seed": 46,
      "full427_max_probability_error": 0.0
    },
    {
      "kind": "plain",
      "seed": 47,
      "full427_max_probability_error": 0.0
    },
    {
      "kind": "moments",
      "seed": 45,
      "full427_max_probability_error": 0.0
    },
    {
      "kind": "moments",
      "seed": 46,
      "full427_max_probability_error": 0.0
    },
    {
      "kind": "moments",
      "seed": 47,
      "full427_max_probability_error": 0.0
    }
  ],
  "real_neural_training_started": true,
  "overall_research_complete": false
}

任务整体未完成；普通阴性继续依据机制调整，不能据成熟TTA、接近阈值或12项benchmark自停。
