from ultralytics import YOLO
import cv2

# 1. Tayyor modelni yuklash (agar fayl bo'lmasa, avtomatik yuklab oladi)
model = YOLO("yolov8n.pt")

# 2. Veb-kamerani ochish
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Kadrda ob'ektlarni aniqlash
    results = model(frame)

    # 4. Natijalarni ko'rsatish (odamlar avtomatik ravishda ramkalanadi)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8n Real-Time People Detection", annotated_frame)

    # 'q' tugmasini bossangiz chiqish
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()