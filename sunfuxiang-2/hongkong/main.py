from ultralytics import YOLO

model=(YOLO('/data/cle/sunfuxiang-2/hongkong/ultralytics/cfg/models/v8/hongkong/all.yaml'))

model.train(**{'cfg':'ultralytics/cfg/default.yaml'})