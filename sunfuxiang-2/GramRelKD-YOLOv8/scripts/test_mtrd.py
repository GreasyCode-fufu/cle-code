from ultralytics import YOLO

model=(YOLO('/data/cle/sunfuxiang-2/ultralytics-main/runs/detect/hongkong_SR1_P4Only_MTRD/weights/best.pt'))
model.val(**{'cfg':'ultralytics/cfg/default.yaml'})
