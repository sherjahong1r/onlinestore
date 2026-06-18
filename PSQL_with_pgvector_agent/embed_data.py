"""
embed_data.py
-------------
Bu skript bazadagi `products` va `company_embeddings` jadvallaridagi matn
ustunlaridan Sentence-Transformers yordamida embedding hisoblaydi va
natijani `embedding` ustuniga yozadi.

Faqat bir marta (yoki yangi qator qo'shilganda) ishlatiladi - bu
real-time agent kodi emas, balki bazani "tayyorlash" skripti.

Ishlatish:
    python embed_data.py
"""

import os
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL topilmadi. .env faylida belgilang.")

print("Model yuklanmoqda (birinchi marta internetdan yuklab olinadi, biroz vaqt oladi)...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 o'lchamli vector beradi
# Vazifasi: Bu yerda all-MiniLM-L6-v2 nomli model yuklanadi. Bu model Embedding model hisoblanadi.
# Nima ish qiladi? U ixtiyoriy matnni olib, uni 384 o‘lchamli vektorga (384 ta raqamdan iborat ro‘yxatga) 
# aylantirib beradi. Masalan, "olma" va "meva" so‘zlari vektor makonida bir-biriga yaqin joylashadi.
print("Model tayyor.")

def embed_products(conn):
    """products jadvalidagi name + category asosida embedding hisoblaydi."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, category FROM products WHERE embedding IS NULL;"
        )
# Vazifasi: Bu qator juda muhim. U embedding IS NULL sharti orqali faqat hali vektorga aylantirilmagan yangi mahsulotlarni tanlab oladi.
# Nega? Chunki agar bazangizda 1 million qator bo‘lsa, har safar skriptni ishlatganda ularni qayta-qayta hisoblab o‘tirmaydi, faqat yangilarini "vektorlashtiradi".
        rows = cur.fetchall()

    if not rows:
        print("products: yangilanishi kerak bo'lgan qator topilmadi.")
        return

    print(f"products: {len(rows)} qator uchun embedding hisoblanmoqda...")

    for row in rows:
        text = f"{row['name']} - {row['category'] or ''}".strip() # Mahsulot nomi va kategoriyasini birlashtirib bitta matn yasaydi
        vector = model.encode(text).tolist() # model.encode(text) orqali uni raqamli vektorga o‘giradi.

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE products SET embedding = %s WHERE id = %s;", # UPDATE buyrug‘i bilan o‘sha mahsulotning embedding ustuniga shu vektorlarni yozib qo‘yadi.
                (vector, row["id"]),
            )
    conn.commit()
    print(f"products: {len(rows)} qator muvaffaqiyatli yangilandi.")


def embed_company_embeddings(conn):
    """company_embeddings jadvalidagi content ustuni asosida embedding hisoblaydi."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, content FROM company_embeddings WHERE embedding IS NULL AND content IS NOT NULL;"
        )
        rows = cur.fetchall()

    if not rows:
        print("company_embeddings: yangilanishi kerak bo'lgan qator topilmadi.")
        return

    print(f"company_embeddings: {len(rows)} qator uchun embedding hisoblanmoqda...")

    for row in rows:
        text = row["content"]
        vector = model.encode(text).tolist()

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE company_embeddings SET embedding = %s WHERE id = %s;",
                (vector, row["id"]),
            )
    conn.commit()
    print(f"company_embeddings: {len(rows)} qator muvaffaqiyatli yangilandi.")


def main():
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        embed_products(conn)
        embed_company_embeddings(conn)
    print("\nTayyor! Barcha embedding'lar bazaga yozildi.")


if __name__ == "__main__":
    main()


# U faqat bir marta yoki yangi ma'lumot qo'shilganda ishlaydi. Oddiy matnli ma'lumotlarni (masalan, mahsulot nomini) AI tushunadigan vektorlarga (sonlar qatoriga) o‘girib, bazadagi maxsus embedding ustuniga yozib chiqadi.
# Analogi: Bu kutubxonachiga o‘xshaydi, u kitoblarni (matnlarni) indekslab, ularni topish oson bo‘lishi uchun raqamli kartotekaga joylashtirib chiqadi.