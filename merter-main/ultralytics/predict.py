

from ultralytics import YOLO

model = YOLO('./runs/detect/train102/weights/best.pt')

results = model('../video/4.mp4', conf=0.05, device=0, save=True,show=True, save_crop=True)
