import os
import requests
from groq import Groq

# 1. API kalit va Groq mijozini sozlaymiz
os.environ["GROQ_API_KEY"] = "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"
client = Groq()
# API kalitni o'rnatish va Groq mijozini yaratish. Bu bizga Llama 3.3 modeliga so'rov yuborish imkonini beradi.

# 2. Internetdan jonli ob-havoni olib keladigan sodda funksiya
def get_weather_data(city: str) -> str:
    try:
        # Shahar koordinatasini topamiz
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url).json() # Geocoding API'si orqali shahar nomidan uning koordinatalarini olish. Agar shahar topilmasa, foydalanuvchiga xabar beriladi.
        
        if not geo_res.get('results'): 
            return f"Kechirasiz, '{city}' nomli shahar topilmadi."
        
        lat = geo_res['results'][0]['latitude'] 
        lon = geo_res['results'][0]['longitude']
        # Koordinatalar asosida ob-havo ma'lumotlarini olish uchun Open-Meteo API'siga so'rov yuboriladi.
        
        # Koordinata orqali ob-havo ma'lumotini olamiz
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_res = requests.get(weather_url).json()
        # Ob-havo ma'lumotlari muvaffaqiyatli olingan taqdirda, foydalanuvchiga shahar nomi, hozirgi harorat va shamol tezligini chiqaradi. 
        
        current = weather_res['current_weather'] 
        # current_weather qiymati URL-dan olinmaydi.
        # Keyin requests.get(weather_url).json() JavaScript objekti (weather_res) qaytadi va undagi weather_res['current_weather'] maydoni olinadi.
        return f"Harorat: {current['temperature']}°C, Shamol tezligi: {current['windspeed']} km/h"
    except Exception as e:
    # as e esa xatolik ob’yektini e nomli o‘zgaruvchiga beradi.    
        return f"Xatolik yuz berdi: {str(e)}"

# 3. Dasturni ishga tushirish qismi
if __name__ == "__main__":
    
    print("\n--- OB-HAVO TIZIMI ---")
    shahar = input("Istalgan davlat shaharlari nomini kiriting:" + "\n" + "Ob-havo malumoti beriladi (Masalan: Toshkent): ")
    
    
    if shahar.strip():
        # Foydalanuvchi shahar nomini kiritgan bo'lsa, davom etamiz. strip() funksiyasi foydalanuvchi kiritgan matndan bosh va oxiridagi bo'sh joylarni olib tashlaydi. Agar foydalanuvchi faqat bo'sh joy kiritsa, bu shart bajarilmaydi.
        print("\nMa'lumot olinmoqda...")
        # Internetdan quruq ma'lumotni olamiz
        api_natijasi = get_weather_data(shahar)
        
        # Llama-3.3 modeliga ma'lumotni yuborib, o'zbekcha chiroyli matn yasatib olamiz
        response = client.chat.completions.create(
            # chat.completions.creat Groq kutubxonasining maxsus funksiyasi bo'lib, "Modelga matn yuborish va undan javob generatsiya qilish (to'ldirish)" buyrug'ini beradi.
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Siz faqat ob-havo ma'lumotini o'zbek tilida chiroyli shaklda tushuntirib beruvchi yordamchisiz. Ortiqcha mavzularda gapirmang."},
                {"role": "user", "content": f"{shahar} shahri uchun ob-havo ma'lumotlari keldi: {api_natijasi}. Shuni chiroyli o'zbekcha tavsiya ko'rinishida yozib ber."}
            ],
            temperature=0.3
            # temperature parametri modelning javoblaridagi ijodkorlik darajasini boshqaradi. Pastroq qiymat (masalan, 0.3) modelning javoblarini aniq va kamroq tasodifiy qiladi, bu esa ob-havo ma'lumotlarini tushuntirish uchun ideal hisoblanadi. Yuqori qiymat (masalan, 0.7 yoki 1.0) esa modelga ko'proq ijodkorlik va tasodifiylik beradi, lekin bu holatda ob-havo ma'lumotlarini tushuntirishda noaniqliklar paydo bo'lishi mumkin.
        )
        
        print(f"\nNatija: {response.choices[0].message.content}")