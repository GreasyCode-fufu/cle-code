# Independent relation-KD implementation report

## Outcome

Stage 2 is implemented as an external Ultralytics 8.4.84 extension. It does not modify core Ultralytics source, does not use the repository's existing `DistillationModel`, and does not launch the requested 300-epoch formal run.

Final verification:

- relation-loss unit tests: **10 passed**
- real-batch pipeline test: **1 passed**
- combined final regression: **11 passed, 1 deterministic-kernel warning**
- 1-epoch/2%-data smoke: **SUCCESS**
- 20-batch forward-only calibration: **SUCCESS**

## Design

Feature hooks capture layers 15, 18 and 21 once per Teacher/Student forward. Teacher features are detached immediately; Student features keep their graph until backward. Every batch begins by clearing stale references, and the training loop clears them in a `finally` block after backward. All handles are removed when training closes.

For each feature `F`:

1. adaptive-average-pool to `20 x 20`;
2. flatten spatial dimensions to `C x 400`;
3. optionally center each channel;
4. L2-normalize over the spatial dimension;
5. form `G = F F^T`, a per-sample `C x C` channel Gram matrix;
6. compare Student and detached Teacher Gram matrices in float32.

The three scale losses are averaged with configurable non-negative weights. The module has zero trainable parameters. It never constructs an `HW x HW` matrix.

Ultralytics' detection loss vector is already batch-scaled and is reduced with `.sum()`. The KD term therefore uses:

`total_loss = det_loss.sum() + kd_raw * kd_weight * kd_ramp * batch_size`

The KD ramp grows linearly over the configured 10 epochs. AMP remains enabled for model forward; relation/Gram math is forced to float32.

## Integration and isolation

`KDDetectionTrainer` first completes the normal Student setup, then separately loads and freezes the Teacher. It preserves the local 8.4.84 optimizer, EMA, scheduler, callbacks, validation and checkpoint flow while replacing only the forward/loss block.

Verified invariants:

- Teacher `requires_grad=False` for every parameter;
- Teacher stays in eval mode;
- Teacher/optimizer parameter intersection is zero;
- Teacher/Student parameter-object intersection is zero;
- Teacher is absent from EMA, Student state_dict and Student checkpoint;
- Teacher module-state SHA256 is unchanged before/after backward, smoke training and cleanup;
- Student parameters and EMA do change;
- hook trigger count is exactly one per scale per forward;
- feature references and hook handles are released;
- relation loss contains no parameters.

## Real-batch diagnostics

The pipeline test used batch size 4 at 640 px:

| Layer | Feature shape |
|---|---|
| 15 / P3 | `[4, 64, 80, 80]` |
| 18 / P4 | `[4, 96, 40, 40]` |
| 21 / P5 | `[4, 256, 20, 20]` |

Observed values:

- detection loss: `51.71422195`
- raw KD loss: `14.33942795`
- ramp: `0.00588235`
- scaled KD contribution: `0.33739832`
- total loss: `52.05162048`
- peak CUDA allocation: `511.34 MiB`

The smoke run completed 17 training batches. Its epoch means were:

- detection loss: `48.38436620`
- raw KD: `13.28023995`
- scaled KD: `2.52853630`
- scaled KD/detection: `0.05337934`
- P3/P4/P5 relation loss: `10.87572176 / 11.61150747 / 17.35349044`
- Teacher forward: `8.07 ms`; relation loss: `2.17 ms`
- peak CUDA allocation: `565.77 MiB`

## KD-weight calibration

Calibration used 20 real batches, batch size 32, random Student YAML initialization, AMP forward, and no backward or optimizer step.

- median detection loss: `402.96893311`
- median raw KD loss: `14.48543787`
- median batch-aligned raw KD/detection ratio: `1.15832338`

Recommended clipped base weights:

| Target KD/detection ratio | `kd_weight` |
|---|---:|
| 2.5% | `0.0215829192` |
| 5% | `0.0431658384` |
| 10% | `0.0863316768` |

The preferred starting point is **`kd_weight=0.0431658384`** for a nominal 5% contribution before warmup effects. The smoke used `1.0` with a 0.1 ramp and empirically produced about 5.34% for that short run; this does not replace the 20-batch batch-32 calibration.

## Formal launch command (not executed)

```bash
cd /data/cle/sunfuxiang-2/ultralytics-main
/home/login/miniconda3/envs/yolo/bin/python scripts/distill_train.py \
  --mode train \
  --kd-weight 0.0431658384
```

The default formal configuration is 300 epochs, batch 32, 640 px, workers 8, device 0, optimizer auto, AMP on, seed 0, mosaic 1.0, mixup 0, close-mosaic 10, random Student YAML initialization and Teacher checkpoint fixed to the inspected path.

## Constraints and warning

- Supported only on Ultralytics 8.4.84 and one GPU; version mismatch, DDP and compile fail fast.
- PyTorch 2.4.1 warns that CUDA adaptive-average-pool backward lacks a deterministic implementation. This checkout uses deterministic `warn_only`; all gradients and tests remained finite.
- `pytest 9.1.1` was installed into the specified `yolo` environment solely to execute the requested tests; no dependency manifest was edited.
