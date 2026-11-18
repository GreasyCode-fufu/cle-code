from ultralytics import YOLO

model=(YOLO('/home/login/cle/sunfuxiang-1/demo/ours/runs/detect/train48/weights/best.pt'))
model.val(**{'cfg':'ultralytics/cfg/default.yaml'})