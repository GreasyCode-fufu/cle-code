# SR1-P4Only relation-KD training behavior

## Result

Training completed all 300 epochs successfully with a fixed `kd_weight=0.0431658384`.
No dynamic KD-weight adjustment was made. The best validation epoch is **300**
with mAP50-95 **0.43553**.

## Required epoch nodes

| Node | Ramp | Detection | Raw KD | Scaled KD | KD/Det | P3 | P4 | P5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.100 | 396.5503 | 14.4115 | 0.9779 | 0.2482% | 10.8257 | 11.8077 | 20.6012 |
| 5 | 0.500 | 259.1651 | 6.4788 | 4.0013 | 1.5455% | 5.2764 | 5.4175 | 8.7425 |
| 10 | 1.000 | 215.1503 | 3.9941 | 5.2214 | 2.4286% | 3.0538 | 3.0420 | 5.8864 |
| 20 | 1.000 | 187.2087 | 3.2088 | 4.4150 | 2.3608% | 2.3263 | 2.3585 | 4.9417 |
| 50 | 1.000 | 157.7743 | 2.7443 | 3.7758 | 2.3966% | 1.8521 | 1.9521 | 4.4287 |
| 100 | 1.000 | 134.2647 | 2.5192 | 3.4664 | 2.5858% | 1.6917 | 1.7960 | 4.0698 |
| 150 | 1.000 | 121.6335 | 2.3757 | 3.2683 | 2.6915% | 1.6026 | 1.7022 | 3.8223 |
| 200 | 1.000 | 113.9703 | 2.2819 | 3.1399 | 2.7604% | 1.5466 | 1.6500 | 3.6490 |
| 250 | 1.000 | 105.6684 | 2.1997 | 3.0267 | 2.8687% | 1.4999 | 1.6016 | 3.4975 |
| 300 | 1.000 | 89.4469 | 2.6345 | 3.6244 | 4.0674% | 1.8823 | 1.9241 | 4.0972 |

Best epoch is epoch 300, so the epoch-300 row is also the best-epoch row.

Detection loss is the Ultralytics batch-scaled scalar for batch size 32. It is not a
per-image loss; divide by 32 for an approximate per-image composite.

## Post-warmup KD ratio

- epoch range: 10-300 inclusive (291 epochs)
- mean: 0.027070 (2.71%)
- median: 0.026953 (2.70%)
- minimum: 0.023336 (2.33%)
- maximum: 0.041095 (4.11%)
- close-mosaic epochs 291-300 mean/median: 4.08% /
  4.07%

The ratio was **not chronically near 5%**. It remained mostly around 2.3-3.0%, then
approached roughly 4.1% after mosaic was disabled for the final 10 epochs. This is
expected under a fixed weight because Student/Teacher relation differences and the
detection-loss scale evolve during training.

## Scale behavior and convergence

- post-warmup mean P3/P4/P5 relation loss:
  1.693465 /
  1.785026 /
  3.952000
- largest scale by mean: **P5**
- largest-scale counts over all epochs: {'p3_relation_loss': 0, 'p4_relation_loss': 0, 'p5_relation_loss': 300}
- raw KD epoch 10 -> 290: 3.994083 -> 2.150176
  (46.17% decrease)
- raw KD epoch 10 -> 300: 3.994083 -> 2.634512
  (34.04% decrease)

KD loss showed a stable long-term decrease through epoch 290. The final 10
close-mosaic epochs changed the batch distribution and produced a controlled rebound,
but the final raw KD remained below epoch 10. P5 was the largest relation mismatch at
every epoch.

## Time and memory overhead

- post-warmup Teacher forward mean: 9.6633 ms/batch
- post-warmup relation-loss mean: 2.1278 ms/batch
- direct measured KD forward/relation work: 11.7911 ms/batch
- estimated direct forward/relation total: 368.0 s
- original SR1-P4Only `results.csv` time: 2965.70 s
- relation-KD `results.csv` time: 3504.71 s
- observed increase: 539.01 s (18.17%)
- peak CUDA allocation recorded by KD metrics: 3840.62 MiB

The observed training-time increase is larger than Teacher-forward plus relation-loss
timing alone because it also includes KD backward propagation and synchronization.
Epoch peak allocation fluctuated around a stable plateau and did not grow
monotonically.
