import os 
import psycopg2 # PostgreSQL bilan bog‘lanish uchun kutubxona
from dotenv import load_dotenv # .env faylidan atrof-muhit o'zgaruvchilarini yuklash uchun kutubxona

load_dotenv() # .env faylidan atrof-muhit o'zgaruvchilarini yuklash
DB_URL = os.getenv("DB_URL") # .env faylida saqlangan ma’lumotlar: GROQ_API_KEY va DB_URL.

# get_db_schema(): Bu funksiya bazadagi jadvallar va ularning ustunlarini "o‘qib" oladi. 
# Bu agentga bazada qanday jadvallar borligini tushunishga yordam beradi.
def get_db_schema() -> str:
    """LLM bazadagi jadvallar qanday tuzilganini ko'rishi uchun sxemani matn holatida qaytaradi"""
    schema = """ # Baza sxemasi: Talabalar, Fanlar, Baholar. Har bir jadvalning ustunlari va ularning turlari quyidagicha:
    Jadvallar tuzilishi:
    
    1. Table: talabalar
       - id: SERIAL PRIMARY KEY 
       - ism_sharif: VARCHAR(100)
       - akademik_guruh: VARCHAR(20)
       - reyting_bal: INT
       
    2. Table: fanlar
       - id: SERIAL PRIMARY KEY
       - fan_nomi: VARCHAR(100)
       - ustoz_ism: VARCHAR(50)
       
    3. Table: baholar
       - id: SERIAL PRIMARY KEY
       - talaba_id: INT (REFERENCES talabalar.id)
       - fan_id: INT (REFERENCES fanlar.id)
       - baho: INT (1 dan 5 gacha)
       - sana: DATE
    """
    return schema # Bu yerda haqiqiy sxema qaytishi shart, bu faqat misol uchun yozilgan matn.


# execute_sql(): Bu funksiya agent tuzib bergan SQL so‘rovni bazada bajaradi va natijani (javobni) olib qaytaradi.
def execute_sql(query: str): 
    """LLM tomonidan generatsiya qilingan SQL kodini PostgreSQL-da ishga tushiradi"""
    try:
        conn = psycopg2.connect(DB_URL) # DB_URL .env faylida saqlangan ma’lumotlar: GROQ_API_KEY va DB_URL. Bu ma’lumotlar agentning ishlashi uchun zarur bo‘lgan API kaliti va ma’lumot
        cursor = conn.cursor() # PostgreSQL bilan bog‘lanish va so‘rovlarni bajarish uchun kursor yaratamiz
        cursor.execute(query)  # LLM tomonidan generatsiya qilingan SQL kodini PostgreSQL-da ishga tushiramiz
        
        # Agar so'rov SELECT bo'lsa natijani qaytaramiz
        if query.strip().upper().startswith("SELECT"): # SELECT so‘rovlari natija qaytaradi, shuning uchun natijani olish uchun fetchall() metodidan foydalanamiz
            # startswith("SELECT"): So'rov SELECT so'zi bilan boshlanayotganini tekshiradi. Agar shunday bo'lsa, demak, bazadan ma'lumot o'qish kerak.
            result = cursor.fetchall() # fetchall() metodi bajarilgan so'rovning barcha natijalarini olish uchun ishlatiladi. Bu yerda natija ro'yxat shaklida qaytadi, har bir element jadvaldagi bir qatorni ifodalaydi.
        else:
            conn.commit() # Agar so'rov SELECT bo'lmasa, demak, ma'lumot bazasida o'zgarishlar qilish kerak (masalan, INSERT, UPDATE, DELETE). commit() metodi bu o'zgarishlarni bazaga saqlaydi.
            result = "Muvaffaqiyatli bajarildi."
            
        cursor.close() # close(): Ish tugagach, kursorni va ulanishni yopamiz. Bu kompyuter xotirasini bo'shatadi va xavfsizlik uchun muhim.
        conn.close()
        return result
    except Exception as e:
        return f"SQL Xatolik: {str(e)}"
    





    