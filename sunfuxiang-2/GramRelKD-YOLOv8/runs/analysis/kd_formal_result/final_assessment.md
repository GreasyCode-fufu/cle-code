# Final assessment

## A. SUCCESS

test mAP50-95 is at least 0.429 and the Student fused parameters/GFLOPs do not increase.

Formal independent test result:

- Precision: 0.749849
- Recall: 0.675435
- mAP50: 0.704466
- mAP75: 0.459626
- mAP50-95: **0.435680**
- fused parameters: 2,603,129
- unfused parameters: 2,607,913
- GFLOPs: 7.044096 (displayed as 7.0)

Compared with L1-96, RelKD changes mAP50-95 by
**+0.006680** and mAP50 by **+0.011466**,
while using 38,208 fewer fused parameters. Compared with the original
SR1-P4Only, RelKD changes mAP50-95 by **+0.008680**,
mAP75 by **+0.014626**, and mAP50 by
**+0.006466** with exactly the same fused/unfused parameter
counts and test-time GFLOPs.

Spalling improves strongly to AP50-95
0.582001; crack and moisture are
0.410681 and
0.314357, respectively. The overall
success judgment follows the user's primary threshold even though not every
class improves.

No second KD-weight experiment was started.
