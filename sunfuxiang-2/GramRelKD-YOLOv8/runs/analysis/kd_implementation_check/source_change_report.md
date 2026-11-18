# KD Stage 2 source-change report

Project: `/data/cle/sunfuxiang-2/ultralytics-main`  
Version lock: Ultralytics `8.4.84`  
Repository note: this checkout has no `.git` metadata, so integrity was verified with SHA256.

## Protected project files

The baseline and final SHA256 values are identical for every protected file:

| Path | Baseline/final SHA256 | Result |
|---|---|---|
| `ultralytics/engine/trainer.py` | `9773675c6500940a408c69636e4c9a4fe41590775efbb554a431d0842d85258a` | unchanged |
| `ultralytics/models/yolo/detect/train.py` | `c385c66137ddd1090705bf6cbf7e6169ab1147da274178a4a75613bbd0c0d346` | unchanged |
| `ultralytics/utils/loss.py` | `61f2ff23a8a468d6423c38f7629da8d66e71d5463f43a1cf98ea6f170bfb1019` | unchanged |
| `ultralytics/nn/tasks.py` | `5490d3c1863a57f22d9d52b126f479efd311d3c7a7a99db9a8d2760609980219` | unchanged |
| `ultralytics/nn/distill_model.py` | `899b9f6a8b11cb84486f4a9646e6191587a0bb06415e22ee88bf49dd8acaf7d6` | unchanged and unused |
| `ultralytics/cfg/default.yaml` | `a24163300c2bde6e2c307bc8fc25a8b17f83f0441332e7d992292a5bed47e73f` | unchanged |
| Teacher `best.pt` | `39b033ec1a18ea58dbc21d4e4bbcb2bf7426b84c172d461087416d201b51ef04` | unchanged |
| historical Student `best.pt` | `e5a0fe66313be57ee73c7899c30f954b9efba1ac8ea6ccc224584079feea8a3b` | unchanged |

Only the eight files listed in `created_files.json` were added to source/script/test locations. All other task artifacts were written under `runs/analysis/kd_implementation_check` or the dedicated smoke run directory.

The data loader generated `/data/cle/sunfuxiang-2/datasets/hongkongScale/train/labels.cache` during real-batch checks. Logs showed it was newly created by this task; it was deleted after validation to restore the pre-task dataset state. No image, label, dataset YAML, Teacher weight, or historical Student weight was written.

## Architecture boundary

- The extension subclasses `DetectionTrainer`; no monkey patching or core-file edit is used.
- The existing `DistillationModel` path is explicitly rejected and never instantiated.
- Teacher parameters are frozen, kept out of Student optimizer/EMA/state_dict/checkpoints, and hash-checked before and after use.
- DDP and `torch.compile` are rejected explicitly; this implementation is intentionally single-GPU.
