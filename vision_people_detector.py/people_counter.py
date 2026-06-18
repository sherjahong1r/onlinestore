from ultralytics import YOLO  # YOLO modelini ishlatish uchun kutubxona
import cv2  # OpenCV - rasm/video bilan ishlash uchun

# 1. MODELNI YUKLASH
model = YOLO("yolov8n.pt")  
# "yolov8n.pt" - tayyor o'qitilgan model fayli
# n = nano (eng yengil va tez variant)

# 2. KAMERANI OCHISH
cap = cv2.VideoCapture(0)  
# 0 - kompyuterdagi birinchi veb-kamera
# Agar 0 ishlamasa, 1 yoki 2 ni sinab ko'ring

# 3. CHEKSIZ SIKL (real vaqtda ishlash uchun)
while True:
    # 3.1. Kameradan bir kadr (rasm) o'qish
    ret, frame = cap.read()
    # ret - True/False (kadr o'qildimi?)
    # frame - o'qilgan rasm (numpy array)
    
    if not ret:  # Agar kadr o'qilmasa
        break    # Sikldan chiqish
    
    # 3.2. Model orqali kadrda ob'ektlarni aniqlash
    results = model(frame)
    # results - barcha aniqlangan ob'ektlar ro'yxati
    
    # 3.3. Faqat ODAAMLARNI filtrlash va sanash
    people_count = 0
    for r in results:           # Har bir natija uchun
        boxes = r.boxes         # Aniqlangan ob'ektlar (to'rtburchaklar)
        if boxes is not None:   # Agar ob'ektlar bo'lsa
            for box in boxes:   # Har bir ob'ekt uchun
                if int(box.cls[0]) == 0:  # 0 = "person" klassi
                    people_count += 1     # Odamlar sonini oshirish
    
    # 3.4. Ekranga ODAAMLAR SONINI yozish
    cv2.putText(frame, f"People: {people_count}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # (10, 50) - matn joylashuvi (x, y)
    # 1 - shrift o'lchami
    # (0, 255, 0) - yashil rang (BGR formatida)
    # 2 - qalinlik
    
    # 3.5. Aniqlangan ob'ektlar bilan ramkani ko'rsatish
    annotated_frame = results[0].plot()
    # results[0].plot() - barcha aniqlangan ob'ektlarni
    # to'rtburchak va nomlari bilan chizadi
    
    cv2.imshow("People Counter", annotated_frame)
    # Yangi oyna ochib, natijani ko'rsatish
    
    # 3.6. Chiqish tugmasi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # 1ms kutish, agar 'q' tugmasi bosilsa
        break  # Sikldan chiqish

# 4. RESURSLARNI O'CHIRISH
cap.release()      # Kamerani bo'shatish
cv2.destroyAllWindows()  # Barcha oynalarni yopish

















