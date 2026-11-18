from ultralytics import YOLO

model=(YOLO('/data/cle/sunfuxiang-2/ultralytics-main/runs/detect/train-11/weights/best.pt'))
model.val(**{'cfg':'ultralytics/cfg/default.yaml'})