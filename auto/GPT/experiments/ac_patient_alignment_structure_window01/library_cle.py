"""Retained CLE moment stem with training-only patient/class contrastive projector."""
from .c4_moment_stem import C4MomentStem,counted_flops
from .patient_alignment import PatientAlignmentNet,contrastive_loss
ARMS=['mom_cross','cross_only','mom_supcon','mom_within']
class LibraryNet(PatientAlignmentNet):
    def __init__(self,backbone,kind):super().__init__(backbone,kind,C4MomentStem)
