from ultralytics import YOLO

model=(YOLO('/data/cle/sunfuxiang-1/ours/runs/detect/train71/weights/best.pt'))
model.val(**{'cfg':'ultralytics/cfg/default.yaml'})