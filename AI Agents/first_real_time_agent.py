import os # os moduli operatsion tizimi bilan ishlash uchun kerak bo'ladi, masalan, atrof-muhit o'zgaruvchilarini o'qish uchun.
from dotenv import load_dotenv # dotenv kutubxonasi .env faylidan atrof-muhit o'zgaruvchilarini yuklash uchun ishlatiladi. Bu bizga API kalitlarini va boshqa maxfiy ma'lumotlarni kodda bevosita ko'rsatmasdan saqlash imkonini beradi.
from langchain_groq import ChatGroq # langchain_groq kutubxonasi Groq API'si bilan ishlash uchun maxsus yaratilgan. ChatGroq esa Groq modeliga matn yuborish va javob olish uchun ishlatiladi.
from langchain_community.tools import DuckDuckGoSearchRun
# langchain_community kutubxonasi turli xil foydali vositalarni o'z ichiga oladi. DuckDuckGoSearchRun esa DuckDuckGo qidiruv tizimi orqali internetda jonli ma'lumotlarni qidirish uchun ishlatiladi.
from langchain_core.messages import SystemMessage, HumanMessage # langchain_core kutubxonasi chat xabarlarini yaratish uchun ishlatiladi. SystemMessage va HumanMessage esa chatdagi tizim va foydalanuvchi xabarlarini ifodalaydi.

# 1. .env fayli ichidagi kalitlarni tizim xotirasiga yuklaymiz
load_dotenv() 

# 2. Model va Internet qidiruv vositasini ishga tushiramiz
# Tizim GROQ_API_KEY ni avtomatik ravishda orqa fonda o'qib oladi
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2, #  javoblar tasodifiyligini past darajaga sozlaydi
    groq_api_key="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"
)

search_tool = DuckDuckGoSearchRun()

print("\n==================================================")
print("LANGCHAIN INTERNET BOTI: " + "\n" + "SAVOLLARINGIZ BO'LSA YOZISHINGIZ MUMKIN!" + "\n" + "CHIQISH UCHUN 'exit' DEB YOZING.")
print("==================================================")

while True:
    user_input = input("\nSiz: ")
    if user_input.lower() == 'exit':
        break
    if not user_input.strip(): # Foydalanuvchi faqat bo'sh joy kiritgan bo'lsa, davom etmasdan yangi savolni kutadi.
        print("Iltimos, savol kiriting.")
        continue
        
    print("\n[Bot] Internet qidirilmoqda...")
    try:
        live_data = search_tool.invoke(user_input) # DuckDuckGoSearchRun vositasi yordamida foydalanuvchining savoliga mos keladigan internet ma'lumotlarini qidiradi. Agar qidiruv muvaffaqiyatli bo'lsa, natija live_data o'zgaruvchisiga saqlanadi.
        # invoke — bu model obyektining metodi (funksiyasi). U messages ro‘yxatini modelga yuboradi.
    except Exception: 
        live_data = "Internetdan ma'lumot olishda muammo bo'ldi."
        
    # Aniq vaqti olish uchun datetime modulidan foydalanamiz. 
    from datetime import datetime
    hozirgi_vaqt = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # datetime.now() hozirgi sana va vaqtni oladi, .strftime() esa uni "YYYY-MM-DD HH:MM:SS" formatida matnga aylantiradi. Bu format foydalanuvchiga aniq va o'qilishi oson bo'lgan vaqtni taqdim etadi.

    # 2. Messages massivini toza holatda ochamiz
    messages = [
        SystemMessage(content=(
            f"Siz o'zbek tilida gaplashadigan chatbotsiz. Joriy aniq sana va vaqt: {hozirgi_vaqt}. "
            "Agar foydalanuvchi vaqt yoki sana haqida so'rasa, faqat mana shu berilgan aniq vaqtga tayanib javob bering! "
            "Boshqa umumiy savollar uchun internet faktlaridan foydalaning."
            "Hamda yuqorida suhbatni davom ettirishingiz kerak va uni saqlashingiz kerak."
        )),
        # Foydalanuvchi xabarini qo'shamiz
        HumanMessage(content=f"Internet faktlari: {live_data}\n\nSavol: {user_input}") # Foydalanuvchining savoli va internetdan olingan ma'lumotlarni HumanMessage sifatida messages ro'yxatiga qo'shamiz. Bu xabar modelga yuboriladi va model bu ma'lumotlarni tahlil qilib, foydalanuvchining savoliga javob beradi.
    ]

    # model.invoke(messages) yordamida modelga messages ro‘yxatini yuboramiz va modeldan javob olamiz. Model bu xabarlarni tahlil qiladi va foydalanuvchining savoliga internetdan olingan ma'lumotlar asosida javob beradi.
    response = model.invoke(messages)
    print(f"\nBot: {response.content}")