from ultralytics import YOLO

model=(YOLO('/data/cle/sunfuxiang-2/hongkong/runs/detect/train3/weights/best.pt'))
model.val(**{'cfg':'ultralytics/cfg/default.yaml'})