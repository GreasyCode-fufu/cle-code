from ultralytics import YOLO

model=(YOLO('/data/cle/sunfuxiang-2/ultralytics-main/ultralytics/cfg/models/hongkong/C2f_DSM.yaml'))
model.train(**{'cfg':'ultralytics/cfg/default.yaml'})