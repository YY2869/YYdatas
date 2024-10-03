
from ultralytics import YOLO


if __name__ == '__main__':
    # 加载模型
    model = YOLO("yolov8s.pt")  # 从头开始构建新模型  #训练模型（.pt权重文件）

    # Use the model
    model.train(data="datasets/meter.yaml", batch=24, epochs=200, imgsz=640, workers=16, device=0)  # 训练模型
