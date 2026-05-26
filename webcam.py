import cv2
from ultralytics import YOLO
model=YOLO('yolov8s.pt')
cap=cv2.VideoCapture(0)
if cap is None:
    print("unable to detect webcam")
    exit()
print("press 'q' to exit")
while True:
    ret,Frame=cap.read()
    if not ret:
        break
    results=model(Frame)
    annotated=results[0].plot()
    print("Detected Objects:",len(results[0].boxes))
    for box in results[0].boxes:
            cls=int(box.cls[0])
            conf=float(box.conf[0])
            x1,y1,x2,y2=box.xyxy[0]
            print(f"{model.names[cls]}|Conf:{conf:.2f}|Box:{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}")  
    cv2.imshow('YOLO Detection',annotated)
    if cv2.waitKey(1)& 0xFF==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


    