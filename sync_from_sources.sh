#!/usr/bin/env bash
# cle-code 同步脚本
# 作用: (1) 把 /data/cle 下 4 个源仓库的新提交 subtree 合入 monorepo(保留历史)
#       (2) 把 2 个非 git 项目(auto/prueNetwork, sunfuxiang/classifyML)的代码快照镜像进来
#       (3) 推送到 git@github.com:GreasyCode-fufu/cle-code.git
# 注意: 源仓库里"未提交"的改动不会自动进来 —— 需先在源仓库 commit(或让 ZCode 帮你提交)。
set -euo pipefail
MONO=/data/cle/cle-code
cd "$MONO"
git config user.name  >/dev/null || { git config user.name "sunfuxiang"; git config user.email "307157171@qq.com"; }

# ---- 1) git 源仓库: subtree 增量合并 ----
sync_repo() {  # $1=subdir $2=src $3=branch
  local sub=$1 src=$2 branch=$3
  local rname="src_${sub//\//_}"
  git remote add "$rname" "$src" 2>/dev/null || git remote set-url "$rname" "$src"
  git fetch -q "$rname" "$branch"
  # 无变化早退: 源 tip 的树 == monorepo 中该子目录的树
  local tip_tree sub_tree
  tip_tree=$(git rev-parse "FETCH_HEAD^{tree}")
  sub_tree=$(git rev-parse -q --verify "HEAD:$sub" 2>/dev/null || echo NONE)
  if [ "$tip_tree" = "$sub_tree" ]; then echo "[repo] $sub: 无新提交"; return 0; fi
  git merge -s ours --allow-unrelated-histories --no-commit -q FETCH_HEAD 2>/dev/null || true
  git read-tree -m -u --prefix="$sub/" FETCH_HEAD
  if ! git diff --cached --quiet HEAD -- 2>/dev/null; then
    git commit -q -m "同步: $sub 新提交($(git rev-parse --short FETCH_HEAD))"
    echo "[repo] $sub: 已合入新提交"
  else
    git reset -q --hard HEAD   # 无变化, 清掉 merge 状态
    echo "[repo] $sub: 无新提交"
  fi
}
sync_repo "auto/ZCode/Classification" /data/cle/auto/ZCode/Classification main
sync_repo "auto/GPT"                 /data/cle/auto/GPT                 main
sync_repo "auto/shared"              /data/cle/auto/shared              master
sync_repo "sunfuxiang/Classification" /data/cle/sunfuxiang/Classification main

# ---- 2) 非 git 项目: 过滤镜像(仅代码类文件, 单文件<=2MB; --delete 同步删除) ----
FILTERS=(--exclude='__pycache__/' --exclude='.git/' --max-size=2M
  --include='*/' --include='*.py' --include='*.sh' --include='*.md' --include='*.json'
  --include='*.yaml' --include='*.yml' --include='*.ipynb' --include='*.txt'
  --include='*.csv' --include='*.cfg' --include='*.ini' --include='*.toml'
  --include='Makefile' --include='Dockerfile' --include='.gitignore' --exclude='*')
rsync -a --delete --delete-excluded --prune-empty-dirs "${FILTERS[@]}" \
  /data/cle/auto/prueNetwork/       auto/prueNetwork/
rsync -a --delete --delete-excluded --prune-empty-dirs "${FILTERS[@]}" \
  /data/cle/sunfuxiang/classifyML/  sunfuxiang/classifyML/
mkdir -p sunfuxiang-1 sunfuxiang-2
for d in ours backup demo; do
  rsync -a --delete --delete-excluded --prune-empty-dirs "${FILTERS[@]}" \
    "/data/cle/sunfuxiang-1/$d/" "sunfuxiang-1/$d/"
done
rsync -a "${FILTERS[@]}" /data/cle/sunfuxiang-1/generate_rotation_testset.py sunfuxiang-1/
rsync -a --delete --delete-excluded --prune-empty-dirs "${FILTERS[@]}" \
  /data/cle/sunfuxiang-2/hongkong/          sunfuxiang-2/hongkong/
rsync -a --delete --delete-excluded --prune-empty-dirs "${FILTERS[@]}" \
  /data/cle/sunfuxiang-2/GramRelKD-YOLOv8/  sunfuxiang-2/GramRelKD-YOLOv8/
git add -A auto/prueNetwork sunfuxiang/classifyML sunfuxiang-1 sunfuxiang-2
if ! git diff --cached --quiet HEAD -- 2>/dev/null; then
  git commit -q -m "同步: 非 git 项目代码快照($(date +%F))"
  echo "[snap] 非 git 项目: 已更新"
else
  echo "[snap] 非 git 项目: 无变化"
fi

# ---- 3) 推送 ----
git push -q origin main
echo "[push] 完成: $(git rev-parse --short HEAD) -> github.com/GreasyCode-fufu/cle-code (main)"
