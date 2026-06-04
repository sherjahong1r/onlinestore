import os
import time
import re # re moduli matnni tahlil qilish va kerakli ma'lumotlarni ajratib olish uchun ishlatiladi, masalan, foydalanuvchining savolida raqamlarni topish uchun.
from datetime import datetime
from dotenv import load_dotenv # .env faylidan atrof-muhit o'zgaruvchilarini yuklash uchun kerak bo'ladi, bu yerda API kalitlarni saqlash uchun ishlatiladi.
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage # LangChain kutubxonasidan SystemMessage va HumanMessage klasslarini import qilamiz. Bu klasslar, modelga yuboriladigan xabarlarni formatlash uchun ishlatiladi. SystemMessage odatda modelga beriladigan ko'rsatmalar yoki kontekst uchun ishlatiladi, HumanMessage esa foydalanuvchining savollarini yoki so'rovlarini ifodalash uchun ishlatiladi.

# 1. Atrof-muhit o'zgaruvchilarini yuklaymiz
load_dotenv() # load_dotenv() funksiyasi, joriy ishchi katalogda joylashgan .env faylidan atrof-muhit o'zgaruvchilarini yuklaydi. Bu, API kalitlarni va boshqa maxfiy ma'lumotlarni kodda bevosita yozmasdan saqlash imkonini beradi. Masalan, .env faylida GROQ_API_KEY=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 deb yozilgan bo'lsa, load_dotenv() bu o'zgaruvchini atrof-muhitga yuklaydi va os.environ["GROQ_API_KEY"] orqali unga murojaat qilish mumkin bo'ladi.

# 2. Model va internet qidiruv tizimini E'LON QILAMIZ
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY") # API kalitini .env faylidan yuklaydi
)

search_tool = DuckDuckGoSearchRun()

# 3. Suhbatlar tarixini saqlash uchun cheksiz ombor
history = []

print("\n=======================================================")
print("LANGCHAIN INTERNET BOTI: SAVOLLARINIGIZNI YOZISHINGIZ MUMKIN!")
print("CHIQISH UCHUN 'exit' DEB YOZING.")
print("=======================================================")

# 4. Asosiy dastur tsikli
while True:
    user_input = input("\nSiz: ")
    if user_input.lower() == 'exit':
        break
    if not user_input.strip():
        print("Iltimos, savol kiriting.")
        continue

    hozirgi_vaqt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # DINAMIK XOTIRA FILTRI (Foydalanuvchi tarixni so'raganda)
    if "savol" in user_input.lower() and ("ro'yxat" in user_input.lower() or "tarix" in user_input.lower() or "chiqar" in user_input.lower()):
        if not history:
            print("\n[Bot]: Siz hali hech qanday savol bermadingiz.")
            continue
        
        raqamlar = re.findall(r'\d+', user_input) # Foydalanuvchining savolida raqamlarni qidiradi. Masalan, "Oxirgi 3 ta savolni ko'rsat" deb yozsa, bu yerda 3 raqami topiladi.
        N = int(raqamlar[0]) if raqamlar else 5     
            
        print(f"\n[Bot]: Sizning oxirgi {min(N, len(history))} ta bergan savollaringiz va ularga ketgan vaqt:")
        tanlangan_tarix = history[-N:] # Tarixdan oxirgi N ta savolni tanlaydi. Agar foydalanuvchi 3 ta savolni ko'rsatishni so'rasa, bu yerda oxirgi 3 ta savol tanlanadi. Agar foydalanuvchi raqam kiritmasa, default ravishda oxirgi 5 ta savol tanlanadi.
        
        for i, muloqot in enumerate(tanlangan_tarix, 1): # Tanlangan tarixdagi har bir savol va unga ketgan vaqtni foydalanuvchiga ko'rsatadi. i - bu savolning tartib raqami, muloqot['savol'] - foydalanuvchining savoli, muloqot['sana_vaqt'] - savol berilgan sana va vaqt, muloqot['ketgan_vaqt'] - model javobini generatsiya qilishga ketgan vaqt sekundlarda.
            print(f"{i}. Savol: \"{muloqot['savol']}\" | Vaqti: {muloqot['sana_vaqt']} | Tezligi: {muloqot['ketgan_vaqt']} sekund") # Savollar va ularning javobga ketgan vaqtini formatlangan holda chiqaradi. Masalan: "1. Savol: "Bugun ob-havo qanday?" | Vaqti: 2024-06-01 14:30:00 | Tezligi: 2.5 sekund"
        continue

    # Oddiy savollar uchun jarayon
    print("\n[Bot] Internet qidirilmoqda...")
    try:
        live_data = search_tool.invoke(user_input)
    except Exception:
        live_data = "Internetdan ma'lumot olishda muammo bo'ldi."

    # Sekundomerni ishga tushiramiz
    start_time = time.time()

    messages = [
        SystemMessage(content=(
            f"Siz o'zbek tilida gaplashadigan chatbotsiz. Joriy aniq sana va vaqt: {hozirgi_vaqt}. "
            "Agar foydalanuvchi vaqt yoki sana haqida so'rasa, faqat shu berilgan vaqtga tayanib javob bering! "
            "Boshqa umumiy savollar uchun internet faktlaridan foydalaning."
        ))
    ]
    
    # Kontekstni miyaga yuklash
    for muloqot in history[-10:]: # Tarixdan oxirgi 10 ta savol-javobни tanlaydi. Bu, modelga foydalanuvchining so'nggi savollari va ularga berilgan javoblar haqida kontekst beradi, shunda model foydalanuvchining ehtiyojlarini yaxshiroq tushunishi mumkin.
        messages.append(HumanMessage(content=muloqot['savol'])) # Har bir savolни HumanMessage sifatida messages ro'yxatiga qo'shadi. Bu, modelga foydalanuvchining so'nggi savollari haqida ma'lumot beradi.
        messages.append(muloqot['javob_obyekt']) # Har bir javobни javob obyekt sifatida messages ro'yxatiga qo'shadi. Bu, modelga foydalanuvchining so'nggi savollariga qanday javoblar berilganini ko'rsatadi, shunda model foydalanuvchining ehtiyojlarini yaxshiroq tushunishi mumkin.
     
    messages.append(HumanMessage(content=f"Internet faktlari: {live_data}\n\nSavol: {user_input}")) # Foydalanuvchining yangi savoli va internetdan olingan ma'lumotlarni HumanMessage sifatida messages ro'yxatiga qo'shadi. Bu, modelga foydalanuvchining yangi savoli va unga tegishli internet ma'lumotlari haqida ma'lumot beradi, shunda model foydalanuvchining ehtiyojlarini yaxshiroq tushunishi mumkin.
    
    # Model javobi va vaqtni hisoblash
    response = model.invoke(messages) # model.invoke(messages) yordamida modelga messages ro‘yxatini yuboramiz va modeldan javob olamiz. Model bu xabarlarni tahlil qiladi va foydalanuvchining savoliga internetdan olingan ma'lumotlar asosida javob beradi.
    end_time = time.time() # Javob generatsiya qilinishini tugatgan vaqti. start_time va end_time orasidagi farq model javobini generatsiya qilishga ketgan vaqtni beradi.
    elapsed_time = round(end_time - start_time, 2) # Javob generatsiya qilishga ketgan vaqtni sekundlarda hisoblaydi va uni 2 onlik raqamgacha yaxlitlaydi. Masalan, agar javob generatsiya qilishga 2.34567 sekund ketgan bo'lsa, elapsed_time 2.35 ga teng bo'ladi.
   
    print(f"\n[Bot]: {response.content}") # Modeldan olingan javobни foydalanuvchiga chiqaradi. response.content modelning javob matnни o'z ichiga oladi, shuning uchun bu yerda faqat javob matни ko'rsatiladi.
    print(f"[Ketgan vaqt]: Ushbu javob {elapsed_time} sekundda generatsiya qilindi]")
    
    # Tarixни saqlab borish
    history.append({
        "savol": user_input, 
        "sana_vaqt": hozirgi_vaqt, 
        "ketgan_vaqt": elapsed_time, # Model javobini generatsiya qilishga ketgan vaqtni saqlaydi. Bu, foydalanuvchiga o'z savollariga javob berish uchun modelning qancha vaqt sarflayotganini ko'rsatish imkonini beradi.
        "javob_obyekt": response # Modeldan olingan javob obyektini saqlaydi.
    })