"""Convenient entry point for the copied DSG-Net (direction-structure network)."""
from models.optimize.convnext_cle_dsg import convnext_tiny_cle_oca_strip

def build_model(num_classes=3):
    return convnext_tiny_cle_oca_strip(num_classes=num_classes)
