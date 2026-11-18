# MTRD Source Integrity Report

## Conclusion

Status: **PASS**. Integrity was verified with SHA256 file manifests; the project has no `.git`, so no git-diff claim is made.

## Created source files

- `ultralytics_ext/markov_relation_kd.py` — `1b80531080004fc84f792cba0264c2fd0725b0a08071240ef613ec2cf5212ff5`
- `ultralytics_ext/markov_kd_trainer.py` — `e7ed6191353a877728e9e0ce605175232afef801aba5448e9ccffc3eaa577e90`
- `scripts/markov_distill_train.py` — `f3e64413d39244b9c0416fe38b83c2bab03d84acea1b540dbb11bdfe7bda61af`
- `scripts/calibrate_markov_temperature.py` — `2c1b6c2caff4cf4383da4a54fa3fdc0eb10c4aaf869a50aecbacd8b898b364d0`
- `scripts/calibrate_markov_kd_weight.py` — `ebbe4eba9ab56878a79db3e67b2718fb9a2db04acc5e8ff0f120592ed0963378`
- `scripts/diagnose_markov_gradients.py` — `175d1aa015e4c97349ee03efff69265645d43816b61228d960fb86d8e1f05397`
- `tests/test_markov_relation_kd.py` — `d24791118ebfb0e6b4934cd115a067f8e4009e7cfd819c98a155c63cba1287f3`
- `tests/test_markov_kd_pipeline.py` — `68436179ea932e084732d73a6e89f0adac56dccd6114adc0a8b10c4ad4d58cf6`

## Existing independent KD files

No existing Gram RelKD source file was modified. `relation_kd.py`, `kd_trainer.py`, `distill_train.py`, existing calibration scripts and existing tests retain their baseline SHA256 values.

## Ultralytics core

- Core Python/YAML manifest before: `7b9dd9a13e2ca8f87ac3310725b3c7b88bf4019f058b2971b5fe9957c67b6894`
- Core Python/YAML manifest after: `7b9dd9a13e2ca8f87ac3310725b3c7b88bf4019f058b2971b5fe9957c67b6894`
- Core source changed: **No**

## Protected weights

- Teacher best.pt before/after: `39b033ec1a18ea58dbc21d4e4bbcb2bf7426b84c172d461087416d201b51ef04` — unchanged
- RelKD best.pt: `221304b847e9b900ccbef36723b27e72227d661279d735e131d0c60a41607ef7` — unchanged
- RelKD last.pt: `a5df9b01346829241d41767319b4e97dd0bd5ce6f2640b6866174624fc18e55b` — unchanged
- Paired NoKD best.pt: `6eed9dfa8c01861af6adf89d0b0f352db384795830d655b960d5dd530f68c9df` — unchanged
- Paired NoKD last.pt: `d819707315180cfd34bca7a835e9ec8f89ce4d672f9a5c0e8e53cd56328d8466` — unchanged

## Existing result directories

| Protected directory | Aggregate before | Aggregate after | Result |
|---|---|---|---|
| kd_implementation_check | `c25e9a5ea62178823611e1b69e038aab02d313bbd675cdcc394355f32a569df4` | same | unchanged |
| kd_formal_result | `92fa99543478f8841e6459bbe6ff83d31e8f7ccd171f377272db391b18eca35d` | same | unchanged |
| kd_paired_nokd_result | `6fc637bf148b3627f977fbff5b43226d078b06e433c8a038b99cd492eb43f418` | same | unchanged |
| RelKD formal training | `8d09a746f79f26bc1df01be123f60587075c72ba2161be86f95719ea9b36906e` | same | unchanged |
| Paired NoKD training | `ac7275b981a2e7624e6ffb996314131134888435a1a913baaaa61adf034c0b48` | same | unchanged |
