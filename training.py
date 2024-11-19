from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
model = YOLO("yolo11n.pt")
results = model.train(data="./datasets/hands.yaml", epochs=3)
results = model.val()
results = model("./datasets/hands/test/17303774950.jpg")
