# CPU 过载限流(紧急)
服务器 CPU 负载 118/24核, 12个训练任务的 DataLoader worker 互相踩踏, GPU 饿死在 1%。
新规(立即生效):
1. 全局训练任务上限 8: 各Agent自查 pgrep -cf "train_cle_fai[r]", 超过自家配额(2)先停自己
2. 每Agent默认并发回落到 2; 高吞吐靠"队列排队"而不是"同时挤"
3. zcode 已砍掉 Batch M 减压(gmp/dshortcut 稍后重排)
