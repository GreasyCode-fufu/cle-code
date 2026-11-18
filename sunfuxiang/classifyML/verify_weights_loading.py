#!/usr/bin/env python3
"""
Verify that all pretrained weights can be loaded correctly for training.
Tests model building and weight loading without actual training.
"""
import sys
import torch
from cle_fair_common import build_model

# Configuration
PTH_DIR = "pth"
DEVICE = "cpu"  # Use CPU for verification

weights_map = {
    'fast_t0': 'fasternet_t0-epoch.281-val_acc1.71.9180.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'densenet121': 'densenet121-a639ec97.pth'
}

print("=" * 60)
print("Pretrained Weights Loading Verification")
print("=" * 60)

all_ok = True
for model_name, weight_file in weights_map.items():
    weight_path = f"{PTH_DIR}/{weight_file}"
    print(f"\n[{model_name}]")
    print(f"  Weight file: {weight_file}")

    # Check file exists
    import os
    if not os.path.exists(weight_path):
        print(f"  ❌ File not found!")
        all_ok = False
        continue

    # Build model
    try:
        model = build_model(model_name, num_classes=2)
        print(f"  ✓ Model built successfully")
    except Exception as e:
        print(f"  ❌ Model build failed: {e}")
        all_ok = False
        continue

    # Load weights
    try:
        # ResNet34 uses legacy format, needs weights_only=False
        weights_only_arg = False if model_name == 'resnet34' else True
        state = torch.load(weight_path, map_location=DEVICE, weights_only=weights_only_arg)

        if isinstance(state, dict) and 'model' in state:
            state = state['model']

        # Filter valid keys (exclude classifier head for pretrained weights)
        valid_keys = [k for k in state.keys() if not any(x in k for x in ['head', 'classifier', 'fc'])]
        filtered_state = {k: v for k, v in state.items() if k in valid_keys}

        miss = model.load_state_dict(filtered_state, strict=False)
        print(f"  ✓ Weights loaded successfully")
        print(f"    - Missing keys: {len(miss.missing_keys)} (expected: classifier head)")
        print(f"    - Unexpected keys: {len(miss.unexpected_keys)}")
    except Exception as e:
        print(f"  ❌ Weight loading failed: {e}")
        all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("✓ All weights can be loaded successfully!")
    print("Ready to run training.")
else:
    print("❌ Some weights failed to load. Please check the errors above.")
    sys.exit(1)
print("=" * 60)
