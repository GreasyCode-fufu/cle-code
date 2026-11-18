from ultralytics import YOLO

# Load a model
model = YOLO('/home/u2318483061/swin/ultralytics/cfg/models/v8/classify/CSPPC.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data= '/home/u2318483061/clsData/maize' , imgsz=[224, 224])