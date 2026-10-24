# C4功率形状轮账本

P全12模型/60epoch/主概率/直接Fraction/目标投影审计通过且无新数值达标，完整307包已交付；原Mom保留，aux不作为确认增量。H原型预注册/原始hash故障/独立修复/证明均留存，新正式15模型在CPU/CUDA通过后实测并发并实际开训。

{
  "created_at": "2026-10-25T05:22:09.307357+08:00",
  "source_config_sha256": "39c104371c68e014d4925a7daab9dd7da003f722ea2169bcdf9a802f84eec0a7",
  "CPU_audit_sha256": "b1ae6e12b2dc3af910ded0e71108c7f09598a9887767ce388e7a175fe90909e3",
  "CUDA_audit_sha256": "865d179f32726e31914adea2d4e6cc060c1b9991eb41659298815644f572bc40",
  "comparison_preregistration_sha256": "938e2fe4775d29ab1aca813a2fe17363abaf0113b052a02db79436fa3ade39a7",
  "models": 15,
  "completed": 0,
  "verified_live_workers": [
    {
      "job": "power_mom_shape_s45",
      "pid": 995967,
      "epochs": 1,
      "actual_train_count": 2077,
      "index_set_sha256": "e8fe76bbcf6d8faf8dcb6f38a148341ffe81d10ac0cdc3e4dc38e9f8ccafd691",
      "samples_per_second": 402.0007046675081
    }
  ],
  "profile_policy": {
    "slots_by_arm": {
      "mom_shape": 1,
      "shape_only": 1,
      "mom_permuted": 1,
      "mom_duplicate": 1,
      "mom_rawh2": 1
    },
    "rule": "smallest within97percent of measured best; active-kernel GPU samples, not interpreter startup;6GiB reserve"
  },
  "mean_recent_GPU": 48.25,
  "recent_resource_rows": 4,
  "real_training_started": true,
  "three_seed_effectiveness_not_established": true,
  "overall_research_complete": false
}

整体目标active，不能因本轮或阴性自停。
