
from ultralytics import YOLO

data= 'ultralytics/cfg/datasets/ogsod-1.0.yaml' 
model = YOLO('ultralytics/cfg/models/MoDA/MoDA_Small_OBB.yaml')
project = 'ablation/OGSOD-2.0/'
name = 'dinov3_base_mlas_c2f-bs64-300e'
model.train(data=data, epochs=300, batch=64, imgsz=256, name=name, device=0, project=project)