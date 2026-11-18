from ultralytics import YOLO

model=(YOLO('/home/login/cle/sunfuxiang-1/ours/ultralytics/cfg/models/ours/demo.yaml'))
model.train(**{'cfg':'ultralytics/cfg/default.yaml'})