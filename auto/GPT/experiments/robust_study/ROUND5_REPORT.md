# CLE round 5: noise-adaptive training and patient balance

12 new trainings and 6 reused baseline models. Same plain ResNet34/224; no test evaluation or labels rewritten.
Development-selected exploratory intervals condition on fitted models and do not correct repeated selection/multiplicity.

| Arm | Mean val F1 | Malignant recall | Patient mean accuracy | Delta vs full CE | Patient interval |
|---|---:|---:|---:|---:|---|
| full_ce | 0.871285 | 0.717949 | 0.880922 | +0.000000 | [0.0, 0.0] |
| grade_ce | 0.881958 | 0.775641 | 0.886453 | +0.010673 | [-0.012271072985530443, 0.03520457368252168] |
| label_smooth | 0.866266 | 0.764957 | 0.866544 | -0.005019 | [-0.025919293828630337, 0.015008957975356759] |
| gce | 0.877132 | 0.745726 | 0.872981 | +0.005847 | [-0.012549836408327386, 0.02501063729889625] |
| elr | 0.876379 | 0.803419 | 0.875235 | +0.005094 | [-0.019642337952525134, 0.029091891040816813] |
| patient_balance | 0.857841 | 0.705128 | 0.858757 | -0.013444 | [-0.03175768583738461, 0.0027486227355079723] |