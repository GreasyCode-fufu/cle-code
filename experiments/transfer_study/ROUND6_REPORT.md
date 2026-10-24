# CLE round 6: constrain pretrained adaptation

12 new trainings and 6 reused baseline models. Same plain ResNet34/224. New arms use grade weights; no test or relabeling.
Development-selected exploratory intervals condition on fitted models and do not correct repeated selection/multiplicity.

| Arm | Mean val F1 | Malignant recall | Patient mean accuracy | Delta vs full CE | Patient interval |
|---|---:|---:|---:|---:|---|
| full_ce | 0.871285 | 0.717949 | 0.880922 | +0.000000 | [0.0, 0.0] |
| grade_ce | 0.881958 | 0.775641 | 0.886453 | +0.010673 | [-0.012271072985530443, 0.03520457368252168] |
| head_only | 0.845186 | 0.675214 | 0.867288 | -0.026099 | [-0.06510425470714827, 0.007695499674863287] |
| stage4_only | 0.879003 | 0.754274 | 0.880998 | +0.007718 | [-0.010136836217960986, 0.027687720502065485] |
| bn_frozen | 0.889178 | 0.816239 | 0.883324 | +0.017894 | [-0.002384753730186669, 0.041240860851022895] |
| starting_anchor | 0.882276 | 0.756410 | 0.879875 | +0.010991 | [-0.007514051025715829, 0.034672405084289384] |