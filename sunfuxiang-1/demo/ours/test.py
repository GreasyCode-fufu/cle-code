from ultralytics import YOLO

model=(YOLO('/home/login/cle/sunfuxiang-1/demo/ours/runs/tune_energy_gated_ocspdconv_s/01_base_auto_s_b16/weights/best.pt'))
model.val(**{'cfg':'ultralytics/cfg/default.yaml'})