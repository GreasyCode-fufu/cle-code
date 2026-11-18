  cd /home/login/cle/sunfuxiang/classifyML

  # 1) full 系列（训练于 3class）→ 3class/test
  for m in resnet34 fast_t0 vit_base_patch16_224 swin_tiny_patch4_window7_224 pvt_tiny repvit_m1; do
    python evaluate.py --model "$m" \
      --weights "weights_3class_full_seed44/$m/${m}_3class_full_seed44_best.pth" \
      --data-path /home/login/cle/sunfuxiang/datasets/3class/test
  done

  # 2) hq 系列（训练于 3class_highquality）→ 3class_highquality/test
  for m in resnet34 fast_t0 vit_base_patch16_224 swin_tiny_patch4_window7_224 pvt_tiny repvit_m1; do
    python evaluate.py --model "$m" \
      --weights "weights_3class_hq_seed44/$m/${m}_3class_hq_seed44_best.pth" \
      --data-path /home/login/cle/sunfuxiang/datasets/3class_highquality/test
  done