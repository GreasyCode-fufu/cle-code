from ultralytics import YOLO

model=(YOLO('/data/cle/sunfuxiang-1/ours/ultralytics/cfg/models/ours/EnergyGatedOCSPDConv.yaml'))
model.train(**{'cfg':'ultralytics/cfg/default.yaml'})