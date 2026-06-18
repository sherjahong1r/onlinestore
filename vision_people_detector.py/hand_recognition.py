import cv2
import mediapipe as mp

print("🔄 MediaPipe yuklanmoqda... (0.10.30)")

# 0.10.30 versiya uchun to'g'ri usul
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera 0 ochilmadi, 1 ni sinab ko'ramiz...")
    cap = cv2.VideoCapture(1)
    
if not cap.isOpened():
    print("❌ Kamerani ochib bo'lmadi!")
    exit()
else:
    print("✅ Kamera ishga tushdi!")

def get_finger_status(hand_landmarks):
    fingers = []
    
    # Bosh barmoq
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # 4 ta barmoq
    tips = [8, 12, 16, 20]
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

print("🖐️ Qo'lingizni kameraga ko'rsating!")
print("❌ Chiqish uchun 'q' tugmasini bosing")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Kadr o'qilmadi!")
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            fingers = get_finger_status(hand_landmarks)
            finger_count = sum(fingers)
            
            # Imo-ishoralar
            if finger_count == 5:
                gesture = "✋ STOP"
                color = (0, 0, 255)
            elif finger_count == 2:
                gesture = "✌️ VICTORY"
                color = (0, 255, 0)
            elif finger_count == 1:
                gesture = "☝️ POINT"
                color = (255, 255, 0)
            elif finger_count == 0:
                gesture = "👊 FIST"
                color = (255, 0, 0)
            else:
                gesture = f"👋 {finger_count} fingers"
                color = (255, 255, 255)
            
            # Qo'lni chizish
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Imo-ishorani ko'rsatish
            cv2.putText(frame, gesture, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            # Barmoqlar sonini ham ko'rsatish
            cv2.putText(frame, f"Fingers: {finger_count}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("🔄 Resurslar bo'shatilmoqda...")
cap.release()
cv2.destroyAllWindows()
print("✅ Dastur tugadi!")