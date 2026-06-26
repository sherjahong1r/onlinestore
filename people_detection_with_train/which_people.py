"""
============================================================
WEBCAM ORQALI YUZ VA ODAM SONINI SANASH (YAKUNIY, YAXSHILANGAN VERSIYA)
============================================================
Bu kod:
1. Train qilingan SimpleCNN modelini yuklaydi
2. Kompyuter kamerasidan video oqimini oladi
3. Har kadrni "sliding window" orqali (turli o'lchamdagi oynalar bilan) skanerlaydi
4. Topilgan yuz/odamlarni to'rtburchak bilan belgilaydi
5. Bir xil ob'ektga tegishli ortiqcha qutilarni aqlli ravishda birlashtiradi
6. Umumiy odamlar sonini ekranga chiqaradi

MUHIM ESLATMA: bu sliding-window + oddiy CNN classifier yondashuvi.
U YOLO kabi zamonaviy object detection modellaridan SODDAROQ, shuning
uchun:
  - Juda kichik/uzoq/qisman to'silgan odamlarni har doim topa olmasligi mumkin
  - Ba'zan bir xil rangli devor/fon kabi joylarni xato "person" deb
    belgilashi mumkin (false positive)
Bu modelning TABIIY cheklovi, lekin quyidagi sozlamalar bilan
iloji boricha yaxshi natija olishga harakat qilamiz.
============================================================
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import time


# ============================================================
# 1-QISM: MODEL ARXITEKTURASI
# ============================================================
# MUHIM: bu klass Kaggle'da yozgan SimpleCNN bilan harf-baharf bir xil
# bo'lishi kerak, aks holda saqlangan vaznlar (.pth fayl) to'g'ri yuklanmaydi.

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================
# 2-QISM: SOZLAMALAR (kerak bo'lsa shu yerdan moslashtirasiz)
# ============================================================
PATCH_SIZE = 64
MODEL_PATH = 'simple_cnn_face.pth'   # <-- O'ZINGIZNING FAYL NOMINGIZGA MOSLANG

# Qancha turli o'lchamdagi oyna bilan skanerlash.
# Endi YANA kichik o'lchamlar qo'shildi (uzoqdagi odamlarni topish uchun),
# lekin birlashtirish mantiqi (pastda) ANCHA kuchaytirilgan, shunda
# bitta yaqin odamning turli qismlari to'g'ri birlashtiriladi.
WINDOW_SIZES = [28, 40, 64, 96, 128]

STEP_RATIO = 0.5

# Model qanchalik "ishonchli" bo'lsa, natijani qabul qilamiz.
# YUQORI qiymat -> kamroq xato (false positive), lekin ba'zi haqiqiy
# odamlarni ham o'tkazib yuborish xavfi oshadi.
CONF_THRESHOLD_PERSON = 0.97
CONF_THRESHOLD_FACE = 0.99  # 0.95 dan 0.99 ga oshirildi - false positive kamaytirish uchun

# Qayta ishlov berish uchun kadr qancha kichraytiriladi (tezlik uchun)
PROCESS_WIDTH = 320

# NMS va birlashtirish sozlamalari
NMS_OVERLAP_THRESH = 0.15
MERGE_MIN_OVERLAP_RATIO = 0.15

# Batch hajmi - bir vaqtning o'zida modelga nechta patch beriladi
BATCH_CHUNK = 256


# ============================================================
# 3-QISM: MODELNI YUKLASH
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Ishlatiladigan qurilma: {device}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model fayli topilmadi: '{MODEL_PATH}'. "
        f"Iltimos MODEL_PATH o'zgaruvchisini to'g'ri fayl nomi bilan yangilang, "
        f"va .pth fayl shu .py bilan BIR papkada bo'lishini tekshiring."
    )

model = SimpleCNN(num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"Model muvaffaqiyatli yuklandi: {MODEL_PATH}")


# ============================================================
# 4-QISM: PATCHLARNI BATCH QILIB BASHORAT QILISH
# ============================================================
def predict_patches_batch(patches_bgr_list):
    """
    Bir nechta patchni BIRGALIKDA (batch) modelga beradi - bu
    har birini alohida yuborishdan sezilarli tezroq.

    Qaytaradi: [(label, confidence), ...] - har patch uchun
    """
    if len(patches_bgr_list) == 0:
        return []

    tensors = []
    for patch_bgr in patches_bgr_list:
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)

        # Kichik patchlarni kattalashtirishda INTER_CUBIC ishlatamiz -
        # bu standart usuldan sifatliroq, xira ko'rinishni kamaytiradi
        if patch_rgb.shape[0] < PATCH_SIZE:
            patch_resized = cv2.resize(patch_rgb, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC)
        else:
            patch_resized = cv2.resize(patch_rgb, (PATCH_SIZE, PATCH_SIZE))

        patch_norm = patch_resized.astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(patch_norm).permute(2, 0, 1))

    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        confidences, predicted = torch.max(probs, 1)

    return list(zip(predicted.cpu().numpy(), confidences.cpu().numpy()))


# ============================================================
# 5-QISM: SLIDING WINDOW - BUTUN KADRNI SKANERLASH
# ============================================================
def sliding_window_detect(frame):
    """
    Kadrni turli o'lchamdagi oynachalar bilan skanerlaydi, barcha
    patchlarni yig'ib, BATCH qilib modelga beradi (tezlik uchun).

    Qaytaradi: [(x, y, w, h, label, confidence), ...]
    label: 1=person, 2=face (0=background allaqachon filtrlangan)
    """
    H, W = frame.shape[:2]

    all_patches = []
    all_coords = []

    for win_size in WINDOW_SIZES:
        step = max(1, int(win_size * STEP_RATIO))
        for y in range(0, max(1, H - win_size), step):
            for x in range(0, max(1, W - win_size), step):
                patch = frame[y:y+win_size, x:x+win_size]
                all_patches.append(patch)
                all_coords.append((x, y, win_size, win_size))

    if len(all_patches) == 0:
        return []

    detections = []
    for i in range(0, len(all_patches), BATCH_CHUNK):
        chunk_patches = all_patches[i:i+BATCH_CHUNK]
        chunk_coords = all_coords[i:i+BATCH_CHUNK]

        results = predict_patches_batch(chunk_patches)

        for (label, conf), (x, y, w, h) in zip(results, chunk_coords):
            label = int(label)
            conf = float(conf)

            if label == 1 and conf >= CONF_THRESHOLD_PERSON:
                detections.append((x, y, w, h, label, conf))
            elif label == 2 and conf >= CONF_THRESHOLD_FACE:
                detections.append((x, y, w, h, label, conf))
            # label == 0 (background) -> e'tiborsiz qoldiriladi

    return detections


# ============================================================
# 6-QISM: NMS (Non-Maximum Suppression)
# ============================================================
def non_max_suppression(detections, overlap_thresh=NMS_OVERLAP_THRESH):
    """
    Bir xil ob'ektga tegishli ko'plab to'rtburchaklardan faqat
    ENG ISHONCHLISINI qoldiradi.

    Face va person ALOHIDA-ALOHIDA NMS qilinadi, chunki ular fizik
    jihatdan bir-birining ustida joylasha oladi (yuz tananing bir qismi).
    """
    if len(detections) == 0:
        return []

    unique_labels = set(d[4] for d in detections)
    final_keep = []

    for lbl in unique_labels:
        group = [d for d in detections if d[4] == lbl]

        boxes = np.array([[d[0], d[1], d[0]+d[2], d[1]+d[3]] for d in group])
        scores = np.array([d[5] for d in group])

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            inds = np.where(iou <= overlap_thresh)[0]
            order = order[inds + 1]

        final_keep.extend([group[i] for i in keep])

    return final_keep


# ============================================================
# 7-QISM: YAQIN/BIR-BIRINI QOPLAGAN QUTILARNI BIRLASHTIRISH
# ============================================================
def cluster_and_keep_best(detections, cluster_dist_ratio=1.5, vertical_chain_overlap=0.5):
    """
    IKKI BOSQICHLI BIRLASHTIRISH:

    1-BOSQICH (vertikal zanjir): bitta odamning "bosh", "yelka", "ko'krak"
    qutilari odatda BIR XIL GORIZONTAL JOYDA (x ustunida mos keladi),
    lekin VERTIKAL ravishda ketma-ket joylashadi. Shuning uchun, agar
    ikki quti GORIZONTAL jihatdan sezilarli kesishsa (bir xil "ustun"da)
    VA vertikal jihatdan deyarli tutashgan/yaqin bo'lsa - ularni BIR
    ZANJIRGA birlashtiramiz va ENG KATTA qamrovli (bounding) qutini yasaymiz.

    2-BOSQICH (klasterlash): qolgan qutilarni standart markaz-masofa
    klasterlash bilan guruhlaymiz (uzoqdagi kichik odamlar uchun, ular
    odatda yagona, kichik quti bo'lib qoladi - zanjir kerak emas).
    """
    if len(detections) == 0:
        return []

    def horizontal_overlap_ratio(box1, box2):
        """Ikki quti gorizontal (x o'qi bo'yicha) qancha kesishadi (0-1)"""
        x1, w1 = box1[0], box1[2]
        x2, w2 = box2[0], box2[2]
        ix1, ix2 = max(x1, x2), min(x1+w1, x2+w2)
        if ix2 <= ix1:
            return 0.0
        intersection = ix2 - ix1
        return intersection / min(w1, w2)

    def vertically_close(box1, box2, gap_ratio=0.6):
        """Ikki quti vertikal jihatdan tutashgan/yaqinmi?"""
        y1, h1 = box1[1], box1[3]
        y2, h2 = box2[1], box2[3]
        avg_h = (h1 + h2) / 2
        # Ikki quti orasidagi vertikal bo'shliq
        gap = max(y1, y2) - min(y1 + h1, y2 + h2)
        return gap < avg_h * gap_ratio

    unique_labels = set(d[4] for d in detections)
    final_results = []

    for lbl in unique_labels:
        group = [list(d) for d in detections if d[4] == lbl]

        # ---- 1-BOSQICH: vertikal zanjirlarni birlashtirish ----
        merged = True
        while merged:
            merged = False
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    h_overlap = horizontal_overlap_ratio(group[i], group[j])
                    v_close = vertically_close(group[i], group[j])

                    if h_overlap >= vertical_chain_overlap and v_close:
                        x1, y1, w1, h1 = group[i][:4]
                        x2, y2, w2, h2 = group[j][:4]

                        new_x = min(x1, x2)
                        new_y = min(y1, y2)
                        new_x2 = max(x1+w1, x2+w2)
                        new_y2 = max(y1+h1, y2+h2)
                        new_conf = max(group[i][5], group[j][5])

                        group[i] = [new_x, new_y, new_x2-new_x, new_y2-new_y, lbl, new_conf]
                        del group[j]
                        merged = True
                        break
                if merged:
                    break

        # ---- 2-BOSQICH: qolganlarini markaz-masofa klasterlash ----
        centers = [(d[0] + d[2]/2, d[1] + d[3]/2, (d[2]+d[3])/2) for d in group]
        cluster_id = [-1] * len(group)
        current_cluster = 0

        for i in range(len(group)):
            if cluster_id[i] != -1:
                continue
            cluster_id[i] = current_cluster
            cx1, cy1, size1 = centers[i]

            for j in range(i + 1, len(group)):
                if cluster_id[j] != -1:
                    continue
                cx2, cy2, size2 = centers[j]
                dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                avg_size = (size1 + size2) / 2
                if dist < avg_size * cluster_dist_ratio:
                    cluster_id[j] = current_cluster

            current_cluster += 1

        for c_id in range(current_cluster):
            cluster_boxes = [group[i] for i in range(len(group)) if cluster_id[i] == c_id]
            best_box = max(cluster_boxes, key=lambda d: d[2]*d[3])  # ENG KATTA maydonli
            final_results.append(tuple(best_box))

    return final_results


def count_unique_people(detections, distance_ratio=2.0):
    """
    YAKUNIY ODAMLAR SONINI HISOBLASH.

    "Person" va "Face" qutilarini LABELdan QAT'I NAZAR birgalikda
    klasterlaymiz - chunki bitta odamning yuzi va tanasi fizik
    jihatdan yaqin joylashgan. Har bir klaster = 1 ta odam.

    Bu degani: agar bitta odam uchun FAQAT face topilsa (tana
    qisman to'silgan bo'lsa), yoki FAQAT person topilsa (yuz
    kamera tomonga qaramagan bo'lsa), yoki IKKISI HAM topilsa -
    barcha holatlarda bu "1 odam" deb hisoblanadi.

    distance_ratio: qanchalik "kengroq" qidirish (kattaroq son =
                    ko'proq narsani bitta odamga tegishli deb hisoblaydi)
    """
    if len(detections) == 0:
        return 0

    centers = [(d[0] + d[2]/2, d[1] + d[3]/2, (d[2]+d[3])/2) for d in detections]
    cluster_id = [-1] * len(detections)
    current_cluster = 0

    for i in range(len(detections)):
        if cluster_id[i] != -1:
            continue
        cluster_id[i] = current_cluster
        cx1, cy1, size1 = centers[i]

        for j in range(i + 1, len(detections)):
            if cluster_id[j] != -1:
                continue
            cx2, cy2, size2 = centers[j]
            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            avg_size = (size1 + size2) / 2
            if dist < avg_size * distance_ratio:
                cluster_id[j] = current_cluster

        current_cluster += 1

    return current_cluster


# ============================================================
# 8-QISM: ASOSIY WEBCAM SIKLI
# ============================================================
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("XATO: Kamera ochilmadi. Boshqa kamera indeksini sinab ko'ring (1, 2...)")
        return

    print("Kamera ochildi. Chiqish uchun 'q' tugmasini bosing.")

    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kadr o'qib bo'lmadi, qaytadan urinilmoqda...")
            continue

        h, w = frame.shape[:2]
        scale = PROCESS_WIDTH / w
        small_frame = cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)))

        detections = sliding_window_detect(small_frame)
        detections = cluster_and_keep_best(detections, cluster_dist_ratio=1.5, vertical_chain_overlap=0.5)

        # ---- YAKUNIY ODAMLAR SONINI HISOBLASH ----
        # Person va face qutilarini chizish uchun alohida ko'rib chiqamiz,
        # LEKIN sanoq uchun ularni BIRGALIKDA klasterlaymiz - chunki
        # bitta odamning yuzi VA tanasi alohida-alohida aniqlangan bo'lsa,
        # ular YIGINDIDA emas, BITTA odam sifatida sanalishi kerak.
        # Bundan tashqari, agar FAQAT yuz topilsa (tana topilmasa - masalan
        # tana qisman to'silgan bo'lsa), bu HAM "1 odam" deb hisoblanadi.
        person_count = count_unique_people(detections, distance_ratio=0.6)

        face_count = 0
        box_count = 0

        for (x, y, win_w, win_h, label, conf) in detections:
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            orig_w = int(win_w / scale)
            orig_h = int(win_h / scale)

            if label == 2:
                color = (0, 255, 0)
                text = f"Face {conf:.2f}"
                face_count += 1
            else:
                color = (255, 0, 0)
                text = f"Person {conf:.2f}"
                box_count += 1

            cv2.rectangle(frame, (orig_x, orig_y), (orig_x + orig_w, orig_y + orig_h), color, 2)
            cv2.putText(frame, text, (orig_x, max(15, orig_y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS hisoblash (ixtiyoriy, kuzatish uchun foydali)
        curr_time = time.time()
        if curr_time != prev_time:
            fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        info_text = f"Odamlar soni: {person_count}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Yuz va Odam Aniqlash', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()