import easyocr # EasyOCR obyektini yaratish: o'zbek va ingliz tillarini o'qiydi
import json

reader = easyocr.Reader(['uz', 'en'], gpu=False)
# gpu=False: video kartangizni ishlatmaydi, oddiy protsessorda ishlaydi
results = reader.readtext('taqriiiz-1.png', detail=0)
# Rasm faylidan matnlarni o'qib, 'results' ro'yxatiga (list) joylaydi
# detail=0: bizga faqat matnning o'zi kerak (koordinatalar emas)

# 1. Hamma matnni saqlab qo'yamiz (barchasi uchun)
data = {
    "barcha_matnlar": results,  # Rasmda o'qilgan hamma narsa shu yerda
    "tahlil": {                 # Biz uchun muhim qismlar
        "ism": None,
        "sana": None,
        "hisob_raqami": None,
        "summa": None
    }
}

# 2. Kerakli maydonlarni qidirib topamiz
for line in results:
    if "Ism:" in line:
        data["tahlil"]["ism"] = line.replace("Ism:", "").strip()
    elif "Sana:" in line:
        data["tahlil"]["sana"] = line.replace("Sana:", "").strip()
    elif "Hisob raqami:" in line:
        data["tahlil"]["hisob_raqami"] = line.replace("Hisob raqami:", "").strip()
    elif "som" in line.lower() or "sum" in line.lower():
        data["tahlil"]["summa"] = line.strip()

# JSON obyektini chiroyli matn formatiga o'girish
json_output = json.dumps(data, indent=4, ensure_ascii=False)
print(json_output)

# json.dumps: Dastur ichidagi lug'atni (lug'at) .jsonformatidagi matnga aylantiradi.

# py -m easy_ocr.py
# python easy_ocr.py
