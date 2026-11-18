"""Out-of-tree Ultralytics extensions for relation knowledge distillation."""

from .feature_hooks import FeatureHookBank
from .kd_trainer import KDConfig, KDDetectionTrainer
from .relation_kd import ChannelRelationKDLoss

__all__ = ("ChannelRelationKDLoss", "FeatureHookBank", "KDConfig", "KDDetectionTrainer")
