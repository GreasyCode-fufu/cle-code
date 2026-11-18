python train_cle_fair.py --model resnet34_degconv_cpca_v3 --num_classes 3 --img_size 512 --seed 44 \
  --data-path-train /home/login/cle/sunfuxiang/datasets/3class_highquality/train \
  --data-path-val   /home/login/cle/sunfuxiang/datasets/3class_highquality/val \
  --weights /home/login/cle/sunfuxiang/classifyML/pth/resnet34-333f7ec4.pth \
  --save-dir ./weights --epochs 300