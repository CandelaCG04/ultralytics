from ultralytics import YOLO

# Load a model
model = YOLO("../ultralytics/cfg/models/11/yolo11n.yaml")
model.export(format='onnx')
#results = model.train(data="coco128.yaml", epochs=200, imgsz=640) #train the model
#model.save('yolo11.pt')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Check the results of the model on the COCO8 example dataset
results = model.val(data="coco128.yaml")
print(f"mAP50-95: {results.box.map}")
print(f"\nYOLOv11n Parameters: {count_parameters(model):,}")




