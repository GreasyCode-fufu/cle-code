#!/bin/bash
cd /home/u2518483080/Classification
echo "=== $(date +%H:%M:%S) ==="
echo "train procs: $(pgrep -fc train_cle_fair.py || echo 0)  runner: $(pgrep -fc 'runner.py experiments' || echo 0)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1
uptime | sed 's/.*load/load/'
echo "--- progress ---"
for f in experiments/logs/*.log; do
  name=$(basename "$f" .log)
  last=$(grep -aE "^Epoch [0-9]+" "$f" 2>/dev/null | tail -1 | sed -E 's/train_loss=[0-9.]+, train_acc=[0-9.]+, //; s/, lr=[0-9.]+//')
  printf "%-46s %s\n" "$name" "${last:0:76}"
done | sort
echo "--- done ---"
grep -a "^\[done\]" experiments/n1b1.nohup.log 2>/dev/null
grep -a "ALL DONE" experiments/n1b1.nohup.log 2>/dev/null
