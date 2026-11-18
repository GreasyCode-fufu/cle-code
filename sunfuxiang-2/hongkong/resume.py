from ultralytics import YOLO

# Load a model
model = YOLO('/home/u2318483061/swin/runs/detect/train55/weights/last.pt')  # load a partially trained model

# Resume training
results = model.train(resume=True)
