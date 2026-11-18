from ultralytics import YOLO

model=(YOLO('/home/login/cle/sunfuxiang-1/ours/runs/detect/raspberry25/weights/best.pt'))
model.val(**{'cfg':'ultralytics/cfg/default.yaml'})