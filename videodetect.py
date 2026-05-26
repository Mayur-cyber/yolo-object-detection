import cv2
from ultralytics import YOLO
model=YOLO("yolov8s.pt")
video_path="bikes.mp4"
cap=cv2.VideoCapture(video_path)
if cap is None:
    print("Unable to read video")
    exit()
print("Press 'q' to exit")
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=int(cap.get(cv2.CAP_PROP_FPS))
output=cv2.VideoWriter("output.mp4",
cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
while True:
    ret,frame=cap.read()
    if not ret:
        print("Video finished")
        break
    results = model(frame)
    annotated = results[0].plot()
    print("Detected Objects :", len(results[0].boxes))
    for box in results[0].boxes:
        cls = (int)(box.cls[0])
        conf = (float)(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0]
        print(f"{model.names[cls]} | Conf: {conf:.2f} | Box:{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}")
    cv2.imshow("Yolo detection",annotated)
    output.write(annotated)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()


