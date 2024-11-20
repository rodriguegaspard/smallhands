from ultralytics import YOLO
import cv2

model = YOLO('./models/best.onnx')
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 FPS if FPS is unavailable
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving as .mp4
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)
    cv2.imshow('smallhands', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap.release()
cv2.destroyAllWindows()
