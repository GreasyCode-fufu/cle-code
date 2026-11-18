# Final assessment

**Training status: SUCCESS**  
**Assessment: C. MTRD仅优于NoKD**

MTRD test mAP50-95 is 0.432186: Δ vs Paired NoKD +0.010990, Δ vs Gram RelKD -0.003494. Test mAP75 is 0.457769: Δ vs Gram -0.001857.

Class AP50-95 deltas vs Gram: spalling -0.017185, crack -0.003973, moisture +0.010676.

Fused Params/GFLOPs are 2,603,129/7.044096, unchanged from SR1-P4Only. Teacher/checkpoint/inference/core/baseline integrity status: SUCCESS.

The operational thresholds used to make the qualitative wording reproducible are: “clearly above Gram” ≥ +0.003 mAP50-95; “very small difference” < 0.003, paired with ≥ +0.003 mAP75 or ≥ +0.005 class AP50-95.
