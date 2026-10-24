TRAIN_METADATA=None
def compute_training_loss(model,logits,labels,indices):
    global TRAIN_METADATA
    import torch
    from torch.nn import functional as functional
    from models.optimize.ac_patient_alignment_window01.patient_alignment import contrastive_loss
    if TRAIN_METADATA is None:
        cfg=READ(S/'config.json');records=READ(R/'experiments/ac_dataset_v1/dataset_manifest.json');allowed=set(cfg['indices']['train']);rows=[r for r in records if r['index'] in allowed];groups={g:i for i,g in enumerate(sorted({r['group_id'] for r in rows}))}
        assert len(rows)==2077 and len(groups)==182
        TRAIN_METADATA={r['index']:(r['label'],groups[r['group_id']]) for r in rows}
    ids=indices.tolist();assert all(i in TRAIN_METADATA for i in ids),'nontrain objective metadata access'
    assert labels.detach().cpu().tolist()==[TRAIN_METADATA[i][0] for i in ids]
    ce=functional.cross_entropy(logits,labels)
    if model.kind in ['plain','moments']:
        model.objective_stats=dict(ce_loss=float(ce.detach()),contrastive_loss=0.,valid_anchor_fraction=0.,mean_positive_pairs=0.)
        return ce
    patients=torch.tensor([TRAIN_METADATA[i][1] for i in ids],device=labels.device)
    mode={'mom_cross':'cross','cross_only':'cross','mom_supcon':'all','mom_within':'within'}[model.kind]
    aux,stats=contrastive_loss(model.projected_features(),labels,patients,mode,temperature=.1)
    model.objective_stats=dict(ce_loss=float(ce.detach()),contrastive_loss=float(aux.detach()),**stats)
    return ce+.05*aux

