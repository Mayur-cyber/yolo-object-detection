import cv2
from ultralytics import YOLO
model=YOLO('yolov8s.pt')
img=cv2.imread(r"C:\Users\ishas\OneDrive\Desktop\yolo\000000026089.jpg")
if img is None:
    print('Unable to read image')
    exit()
results=model(img)
annotated=results[0].plot()
print("Detected Objects:",len(results[0].boxes))
for box in results[0].boxes:
    cls=int(box.cls[0])
    conf=float(box.conf[0])
    x1,y1,x2,y2=box.xyxy[0]
    print(f"{model.names[cls]}|Conf:{conf:.2f}|Box:{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}")
cv2.imshow("Yolo Detection",annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('result.jpg',annotated)
print("Result saved")
