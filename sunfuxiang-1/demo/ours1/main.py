from ultralytics import YOLO

model=(YOLO('/home/login/cle/sunfuxiang-1/demo/ours/ultralytics/cfg/models/v8/EnergyGatedOCSPDConv.yaml'))
model.train(**{'cfg':'ultralytics/cfg/default.yaml'})