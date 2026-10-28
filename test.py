from ultralytics import YOLO
 
def main():
    model = YOLO('OGSOD-1.0/MoDA-S.best')
    metrics = model.val(split='test', imgsz=256, device=0, batch=1, project='runs/test/OGSOD-2.0', name='MoDA-S')
    print(f"mAP50: {metrics.box.map50:.6f}")
    print(f"mAP75: {metrics.box.map75:.6f}")
    print(f"mAP50-95: {metrics.box.map:.6f}")

if __name__ == '__main__':
    main()
