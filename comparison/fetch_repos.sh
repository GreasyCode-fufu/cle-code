#!/bin/bash
# 对比实验官方代码库获取脚本(R6, 2026-11-09)。每个仓库浅克隆到 third_party/<名称>/(各自独立的新目录),
# 只作为忠实复现的参考阅读, 不在其中执行代码; 实际对比实验在本仓库统一协议框架内实现(见 README.md)。
# 结果(URL + commit + 许可证)写入 third_party/MANIFEST.tsv。
set -u
cd "$(dirname "$0")"
mkdir -p third_party
M=third_party/MANIFEST.tsv
printf "name\turl\tcommit\tlicense\tsparse_paths\n" > $M
clone () { local name=$1 url=$2; shift 2; local sparse="$*"
  local d=third_party/$name
  if [ -d "$d/.git" ]; then echo "[skip] $name exists"; else
    rm -rf "$d"
    if [ -n "$sparse" ]; then
      git clone --depth 1 --filter=blob:none --sparse "$url" "$d" -q && git -C "$d" sparse-checkout set --no-cone $sparse
    else
      git clone --depth 1 "$url" "$d" -q
    fi || { echo "[FAIL] $name"; printf "%s\t%s\tFAILED\t\t%s\n" "$name" "$url" "$sparse" >> $M; return; }
  fi
  local sha lic; sha=$(git -C "$d" rev-parse HEAD)
  lic=$(ls "$d" | grep -i -m1 -E "^(license|licence|copying)" || echo "none-found")
  printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$url" "$sha" "$lic" "$sparse" >> $M
  echo "[ok] $name $sha ($lic) $(du -sh $d | cut -f1)"
}
# ── 实验 1: 微调策略对比方法 ──
clone LP-FT_transfer_learning        https://github.com/AnanyaKumar/transfer_learning
clone SurgicalFT_surgical-finetuning https://github.com/anniesch/surgical-finetuning
clone L2-SP_TransferLearningClassification https://github.com/thuml/TransferLearningClassification
clone cRT_classifier-balancing       https://github.com/facebookresearch/classifier-balancing
# ── 实验 2 / 对比表: 主干 ──
clone ConvNeXt                       https://github.com/facebookresearch/ConvNeXt
clone StarNet_Rewrite-the-Stars      https://github.com/ma-xu/Rewrite-the-Stars
clone FasterNet                      https://github.com/JierunChen/FasterNet
clone GhostNet_Efficient-AI-Backbones https://github.com/huawei-noah/Efficient-AI-Backbones
clone MobileNetV4_timm               https://github.com/huggingface/pytorch-image-models /timm/models/mobilenetv3.py /timm/models/_efficientnet_blocks.py /timm/models/_efficientnet_builder.py /LICENSE /README.md
clone MobileNetV3_ShuffleNetV2_torchvision https://github.com/pytorch/vision /torchvision/models/mobilenetv3.py /torchvision/models/shufflenetv2.py /LICENSE /README.md
echo "manifest -> $M"
