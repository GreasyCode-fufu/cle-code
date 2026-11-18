#!/bin/bash
# R6 最新轻量网络对比: 官方代码库浅克隆(每个网络一个独立子文件夹), 清单写入 MANIFEST.tsv。
# 代码只作为模型定义来源; 训练/评测在本仓库统一协议内进行(见 README.md)。
set -u
cd "$(dirname "$0")"
M=MANIFEST.tsv
[ -f $M ] || printf "folder\turl\tcommit\tlicense\n" > $M
clone () { local name=$1 url=$2
  if [ -d "$name/.git" ]; then echo "[skip] $name"; return; fi
  git clone --depth 1 -q "$url" "$name" || { echo "[FAIL] $name"; return; }
  local sha lic; sha=$(git -C "$name" rev-parse HEAD); lic=$(ls "$name" | grep -i -m1 -E "^(license|licence|copying)" || echo none-found)
  printf "%s\t%s\t%s\t%s\n" "$name" "$url" "$sha" "$lic" >> $M
  echo "[ok] $name ${sha:0:10} ($lic) $(du -sh $name | cut -f1)"
}
clone LSNet_CVPR2025        https://github.com/THU-MIG/lsnet
clone iFormer_ICLR2025      https://github.com/ChuanyangZheng/iFormer
clone EfficientViM_CVPR2025 https://github.com/mlvlab/EfficientViM
clone MobileMamba_CVPR2025  https://github.com/lewandofskee/MobileMamba
clone SHViT_CVPR2024        https://github.com/ysj9909/SHViT
