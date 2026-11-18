"""Exact original selected network with ONLY its Mom wrapper removed."""
from .midstage_focal import MidstageFocalNet, counted_flops
from .c4_moment_stem import C4MomentStem
ARMS = ['misaligned_only']
class LibraryNet(MidstageFocalNet):
    def __init__(self, backbone, arm):
        assert arm in ['plain', 'mom_only', 'mom_misaligned', *ARMS]
        if arm == 'misaligned_only':
            super().__init__(backbone, 'mom_misaligned')
            original = self.encoder[0][0][0]
            assert isinstance(original, C4MomentStem)
            self.encoder[0][0][0] = original.base
            self.arm = arm
        else:
            super().__init__(backbone, arm)
