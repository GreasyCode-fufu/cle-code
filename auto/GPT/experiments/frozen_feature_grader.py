"""Fit-only patient-cross-fitted linear grading on fixed public DINO features."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

WEIGHTS = {'A': 1., 'B': 1., 'C': .75, 'D': .25, 'E': .10}
GRADER_SEEDS = [42, 43, 44]


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z-z.max(1, keepdims=True)
    p = np.exp(z)
    return p/p.sum(1, keepdims=True)


def fit_head(features, labels, fit, held, path):
    fit, held = np.asarray(fit), np.asarray(held)
    x = features[fit].astype(np.float64)
    mean = x.mean(0)
    scale = np.maximum(x.std(0), 1e-6)
    x = (x-mean)/scale
    classifier = LogisticRegression(C=1., l1_ratio=0., solver='lbfgs', max_iter=2000,
                                    tol=1e-6, fit_intercept=True, class_weight=None)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        classifier.fit(x, labels[fit])
    assert classifier.classes_.tolist() == [0, 1, 2]
    assert classifier.n_iter_.max() < 2000
    coef, intercept = classifier.coef_, classifier.intercept_
    p = softmax(((features[held].astype(np.float64)-mean)/scale)@coef.T+intercept)
    assert np.max(np.abs(p-classifier.predict_proba((features[held]-mean)/scale))) < 1e-12
    train_p = softmax(x@coef.T+intercept)
    residual = train_p-np.eye(3)[labels[fit]]
    gradient = x.T@residual/len(fit)+coef.T/len(fit)
    residual_max = max(float(np.abs(gradient).max()), float(np.abs(residual.mean(0)).max()))
    assert residual_max < 5e-5, residual_max
    with path.open('wb') as f:
        np.savez(f, fit_indices=fit, held_indices=held, mean=mean, scale=scale,
                 coef=coef, intercept=intercept, classes=classifier.classes_,
                 n_iter=classifier.n_iter_, gradient_max=np.array(residual_max))
    return p


def partitions(job, labels, groups):
    active = np.array(job['train'], dtype=np.int64)
    for replicate in GRADER_SEEDS:
        random_state = job['seed']*100000+job['fold']*100+replicate
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=random_state)
        for fold, (a, b) in enumerate(splitter.split(active, labels[active], groups[active])):
            fit, held = active[a], active[b]
            assert set(groups[fit]).isdisjoint(groups[held])
            assert set(labels[fit]) == {0, 1, 2}
            yield replicate, fold, fit, held


def plans(job, records, grades):
    active = np.array(job['train'], dtype=np.int64)
    labels = np.array([r['label'] for r in records])
    w = np.array([WEIGHTS[grades[int(i)]] for i in active])
    shuffled = w.copy()
    rng = np.random.default_rng(20261024+job['seed']*100+job['fold'])
    ac = sorted(int(i) for i in active if grades[int(i)] in 'ABC')
    random_ac = []
    for label in [0, 1, 2]:
        positions = np.flatnonzero(labels[active] == label)
        assert len(positions)
        w[positions] /= w[positions].mean()
        shuffled[positions] = rng.permutation(w[positions])
        n = sum(labels[i] == label for i in ac)
        assert n > 0, 'AC retained no images from a class; stop without changing thresholds'
        random_ac += rng.choice(active[positions], size=n, replace=False).tolist()
        assert np.array_equal(np.sort(shuffled[positions]), np.sort(w[positions]))
        assert abs(w[positions].mean()-1) < 1e-12
    return dict(grade_weighted=dict(train=active.tolist(), sample_weights={str(i): float(v) for i, v in zip(active, w)}),
                shuffled_weighted=dict(train=active.tolist(), sample_weights={str(i): float(v) for i, v in zip(active, shuffled)}),
                ac_only=dict(train=ac, sample_weights={}),
                random_ac=dict(train=sorted(random_ac), sample_weights={}))
