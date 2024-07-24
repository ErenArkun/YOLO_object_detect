from ultralytics import YOLO

# Modeli yükleyin
model = YOLO('yolov8s.pt')

# Data and parameters for training
data_config = 'path/your/data.yaml' # yaml file path
epochs = 10
imgsz = 640
batch_size = 8
project_path = 'path/your/folder/yolo_data' # training and testing data
experiment_name = 'save/folder/name' # folder name where the trained model will be saved

# Model training
model.train(
    data=data_config,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size,
    project=project_path,
    name=experiment_name
)