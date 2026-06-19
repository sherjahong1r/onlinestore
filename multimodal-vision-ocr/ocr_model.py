import os
import sys # Terminal buyruqlarini boshqarish
import json # Natijani JSON formatda yozish uchun
import base64 # Rasmlarni AI tushunadigan matn (kod) holiga keltirish uchun
import mimetypes # # Fayl turini (masalan, png yoki jpg) avtomatik aniqlash uchun
from pathlib import Path # Fayl yo'llari bilan xavfsiz va qulay ishlash uchun

from pdf2image import convert_from_path # PDF2Image kutubxonasi PDF sahifalarini rasmga aylantiradi

poppler_path = r'C:\Library\bin' 
# Poppler dasturining kompyuterdagi manzili (PDF-ni rasmga aylantirish uchun dvigatel)

from groq import Groq

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ---------------------------------------------------------------------------
# SOZLAMALAR
# ---------------------------------------------------------------------------

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" 

# SYSTEM_PROMPT = """\
# Vazifa: Hujjatdagi BARCHA matnni, shu jumladan kichik jumlalar va paragraflarni to'liq o'qi. 
# Hech narsani qisqartirma va hech narsani tashlab ketma.
# Natijani "content" maydoniga joylashtir va FAQAT valid JSON formatda qaytar.

# Bu qism AI qanday yordam belgilovchi "aqliy" qismdir.
SYSTEM_PROMPT = """\
Sen hujjatlardan malumot chiqarib oluvchi OCR yordamchisisan.
Senga rasm beriladi (hujjat sahifasi).

Vazifa:
1. Hujjatdagi barcha muhim matn va malumotlarni diqqat bilan o'qi.
2. Natijani FAQAT JSON formatda qaytar - hech qanday qo'shimcha izoh,
   tushuntirish yoki markdown belgilarisiz.
3. Agar hujjat turi aniq bo'lsa (pasport, invoice, kvitansiya va h.k.),
   shu turga mos maydonlarni ajratib chiqar.
4. Agar biror maydon o'qib bo'lmasa yoki mavjud bo'lmasa, uning qiymatini
   null qilib qo'y, lekin maydonni JSON'dan butunlay o'chirma.
5. Raqamlarni (summalar, sanalar, ID raqamlar) iloji boricha to'g'ri va
   aniq formatda ber.
"""


# Bu funksiya bizga yuborgan fayl PDF bo'lsa, uni AI tushunadigan rasmga o'giradi.
def prepare_image_path(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    
    # Agar PDF bo'lmasa, o'zini qaytaradi
    if suffix != '.pdf':
        return file_path
    
    # Agar PDF bo'lsa, rasmga aylantiramiz
    images = convert_from_path(file_path, dpi=300, poppler_path=poppler_path)
    
    # Birinchi sahifani vaqtincha rasm sifatida saqlaymiz
    temp_image_path = "temp_page.png"
    images[0].save(temp_image_path, "PNG")
    
    # Endi funksiya rasm faylining manzilini qaytaradi
    return temp_image_path

def encode_image_to_base64(image_path: str) -> tuple[str, str]: 
    """Rasm faylni base64 ga o'giradi va mime turini aniqlaydi."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png" # Agar aniqlay olmasa, standart png deb oladi

    with open(image_path, "rb") as f: # Rasmni ikkilik (binary) rejimda ochadi  
        data = base64.b64encode(f.read()).decode("utf-8") # ai tushunishi uchun base64

    return data, mime_type


def clean_json_response(text: str) -> str:
    """Model javobidan ```json kabi belgilarni tozalaydi (agar bo'lsa)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    text = text.replace("```json", "").replace("```", "")
    return text.strip()


def extract_to_json(file_path: str) -> dict:
    """Asosiy funksiya: faylni o'qib, JSON natijani qaytaradi."""

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY muhit o'zgaruvchisi topilmadi. "
            "Avval: export GROQ_API_KEY='sizning_kalitingiz'"
        )

    client = Groq(api_key=api_key)

    image_path = prepare_image_path(file_path)
    image_data, mime_type = encode_image_to_base64(image_path)

    # Vaqtinchalik PDF->rasm faylni tozalaymiz
    if image_path != file_path and Path(image_path).exists():
        cleanup_path = image_path
    else:
        cleanup_path = None

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Shu hujjatdagi malumotlarni JSON formatda chiqarib ber.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        },
                    },
                ],
            },
        ],
        response_format={"type": "json_object"}, # Natija JSON bo'lishi shart
    )

    if cleanup_path:
        os.remove(cleanup_path)

    raw_text = response.choices[0].message.content
    cleaned = clean_json_response(raw_text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("DIQQAT: Model qaytargan javob valid JSON emas.", file=sys.stderr)
        print("Xom javob:\n", raw_text, file=sys.stderr)
        raise e


def main():
    if len(sys.argv) < 2: # Terminaldan fayl nomini qabul qiladi
        print("Ishlatilishi: python ocr_groq.py <fayl_yo'li>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"Xato: fayl topilmadi -> {file_path}")
        sys.exit(1)

    print(f"Fayl o'qilmoqda: {file_path} ...")
    result = extract_to_json(file_path)

    print("\n--- NATIJA (JSON) ---\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    out_path = Path(file_path).with_suffix(".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nNatija saqlandi: {out_path}")

if __name__ == "__main__":
    main()


# $env:GROQ_API_KEY="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"
# py ocr_model.py test_namuna.png    
# JPG, JPEG, PNG, WEBP, BMP, pdf

