from ultralytics import YOLO
from PIL import Image
import os

def resize_image(path, output_size):
    with Image.open(path) as img:
        img = img.resize(output_size)
        # img = img.convert("RGB")
        img.save(path)

def resize_images_in_directory(directory, output_size):
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                resize_image(os.path.join(subdir, filename), output_size)

# The root directory where you want to resize images
root_directory = '/home/u2318483061/clsData/pomegranate/test'

# The output size you want for your images (e.g., 1080x1080)
output_size = (224, 224)

# Resize images in the root directory and all its subdirectories
resize_images_in_directory(root_directory, output_size)

# Load a model
# model = YOLO('yolov8n-cls.pt')  # load an official model
model = YOLO('/home/u2318483061/swin/runs/classify/train17/weights/best.pt')  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.top1   # top1 accuracy
metrics.top5   # top5 accuracy