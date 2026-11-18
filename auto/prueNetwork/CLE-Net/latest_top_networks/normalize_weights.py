"""Normalize official checkpoints without altering any parameter tensor."""
import hashlib
from pathlib import Path
import torch

root = Path("/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/pretrained")
source = root / "mambavision_tiny_1k.pth.tar"
target = root / "mambavision_tiny_1k.normalized.pth"

raw = torch.load(source, map_location="cpu", weights_only=False)
state = raw["state_dict"]
sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
torch.save({
    "model": state,
    "source": "nvidia/MambaVision-T-1K:mambavision_tiny_1k.pth.tar",
    "source_sha256": sha256,
    "normalization": "state_dict field copied verbatim",
}, target)
print(target, target.stat().st_size, sha256)


def remap_checkpoint(src_name, dst_name, remap):
    src = root / src_name
    dst = root / dst_name
    raw = torch.load(src, map_location="cpu", weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    mapped = {remap(k): v for k, v in state.items()}
    source_hash = hashlib.sha256(src.read_bytes()).hexdigest()
    torch.save({
        "model": mapped,
        "source": src_name,
        "source_sha256": source_hash,
        "normalization": "classifier-only head remap; tensors copied verbatim",
    }, dst)
    print(dst, dst.stat().st_size, source_hash)


def mambaout_key(k):
    if k.startswith("head.fc2."):
        return "head." + k[len("head.fc2."):]
    if k.startswith("head."):
        return "pre_head." + k[len("head."):]
    return k


def cpubone_key(k):
    if k.startswith("head.classifier.linear."):
        return "head." + k[len("head.classifier.linear."):]
    if k.startswith("head."):
        return "pre_head." + k[len("head."):]
    return k


remap_checkpoint("mambaout_tiny.pth", "mambaout_tiny.normalized.pth", mambaout_key)
remap_checkpoint(
    "cpubone_b2_bfrobust_r224_in1k.pth",
    "cpubone_b2_bfrobust_r224_in1k.normalized.pth",
    cpubone_key,
)
