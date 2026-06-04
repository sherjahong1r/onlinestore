import os
import requests
import json  # json kutubxonasi, asbob argumentlarini tahlil qilish va natijalarni formatlash uchun kerak bo'ladi.
from groq import Groq
# Groq kompaniyasining rasmiy kutubxonasi. Llama 3.3 modeli bilan to'g'ridan-to'g'ri bog'lanishni ta'minlaydi.

# 1. API kalit va Groq mijozini sozlaymiz
os.environ["GROQ_API_KEY"] = "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"
client = Groq()
# API kalitni o'rnatish va Groq mijozini yaratish. Bu bizga Llama 3.3 modeliga so'rov yuborish imkonini beradi.

# 2. Agent foydalanadigan funksiya (Asbob)
def get_weather(city: str) -> str:
    # get_weather funksiyasi, shahar nomini qabul qiladi va hozirgi ob-havo ma'lumotlarini qaytaradi. Bu funksiya model tomonidan chaqiriladi.
    """Berilgan shahar nomi uchun hozirgi jonli ob-havo ma'lumotlarini qaytaradi."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url).json()
        # Geocoding API'si orqali shahar nomidan uning koordinatalarini olish. Agar shahar topilmasa, foydalanuvchiga xabar beriladi.
        
        if not geo_res.get('results'):
            return f"Kechirasiz, '{city}' nomli shahar topilmadi."
        
        lat = geo_res['results'][0]['latitude']
        lon = geo_res['results'][0]['longitude']
        # Koordinatalar asosida ob-havo ma'lumotlarini olish uchun Open-Meteo API'siga so'rov yuboriladi. Hozirgi harorat va shamol tezligi qaytariladi.
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_res = requests.get(weather_url).json()
        # Ob-havo ma'lumotlari muvaffaqiyatli olingan taqdirda, foydalanuvchiga shahar nomi, hozirgi harorat va shamol tezligini o'zbek tilida chiroyli formatda qaytaradi.
        
        current = weather_res['current_weather']
        return f"{city}da hozirgi harorat: {current['temperature']}°C, Shamol tezligi: {current['windspeed']} km/h."
    except Exception as e:
        return f"Ma'lumot olishda xatolik yuz berdi: {str(e)}"
    # get_weather funksiyasi, shahar nomini qabul qiladi va hozirgi ob-havo ma'lumotlarini qaytaradi. Bu funksiya model tomonidan chaqiriladi.

# Model tushunishi uchun asbob sxemasi (Tool Definition)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Berilgan shahar uchun hozirgi real vaqtdagi ob-havo ma'lumotlarini olib keladi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Shahar nomi, masalan: Toshkent, Samarqand, Andijon",
                    }
                },
                "required": ["city"],
            },
        },
    }
]
# Asbob sxemasi, modelga get_weather funksiyasining qanday ishlashini va qanday argumentlarni qabul qilishini tushuntiradi. Bu modelga asbobni to'g'ri chaqirish imkonini beradi.

def run_agent(user_message):
    messages = [
        {"role": "system", "content": "Siz yordamchi AI agentsiz. Ob-havo haqida so'rashsa, har doim get_weather funksiyasini chaqiring va qaytgan ma'lumot asosida o'zbek tilida chiroyli javob bering."},
        {"role": "user", "content": user_message}
    ]
    # run_agent funksiyasi, foydalanuvchidan kelgan xabarni qabul qiladi va modelga yuborish uchun xabarlar zanjirini yaratadi. Model javobida asbob chaqiruvlari bo'lsa, ularni bajaradi va natijani modelga qaytaradi.

    # 1-Qadam: Modelga so'rov yuborish
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0
    )
    # Modelga so'rov yuboriladi. messages zanjiri, asbob sxemasi va boshqa parametrlar bilan birga, model javobida asbob chaqiruvlari bo'lsa, ularni avtomatik aniqlash uchun tool_choice "auto" ga o'rnatiladi.
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # model javobida asbob chaqiruvlari mavjudligini tekshiramiz. Agar mavjud bo'lsa, ularni bajarish va natijani modelga qaytarish jarayoniga o'tamiz.
    
    # 2-Qadam: Agar model asbobni ishlatishni xohlasa
    if tool_calls:
        print("\n[Agent Fikri]: Ob-havoni bilish uchun tizim asbobini chaqirishim kerak.....")
        
        # Qaysi funksiyani chaqirishni aniqlaymiz
        available_functions = {"get_weather": get_weather}
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            # Asbob chaqiruvini bajarish uchun kerakli argumentlarni olish va funksiya chaqirish. Bu yerda, model get_weather funksiyasini chaqirishni xohlasa, biz uni bajarib, natijani olamiz.

            # Funksiyani chaqiramiz (API'dan jonli ma'lumot keladi)
            tool_output = function_to_call(city=function_args.get("city"))
            print(f"[Asbob Natijasi]: {tool_output}")
            # Asbobdan olingan natijani modelga qaytarish uchun xabarlar zanjiriga qo'shamiz. Bu natija model tomonidan yakuniy javobni yaratishda ishlatiladi.
            
            # Natijani xabarlar zanjiriga qo'shamiz
            messages.append(response_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": tool_output,
            })
            # Asbobdan olingan natijani modelga qaytarish uchun xabarlar zanjiriga qo'shamiz. Bu natija model tomonidan yakuniy javobni yaratishda ishlatiladi.
        
        # 3-Qadam: Modelga natijani qaytarib, yakuniy javobni olamiz
        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
        )
        return final_response.choices[0].message.content
    
    return response_message.content
# run_agent funksiyasi, foydalanuvchidan kelgan xabarni qabul qiladi va modelga yuborish uchun xabarlar zanjirini yaratadi. Model javobida asbob chaqiruvlari bo'lsa, ularni bajaradi va natijani modelga qaytaradi.

# Terminal Interfeysi
if __name__ == "__main__":
    print("\n==================================================")
    print("AI Agent tayyor! Savolingizni bering (chiqish uchun 'exit' deb yozing):")
    print("Men real vaqtli ob-havo ma'lumotlarini ham beraman!")
    print("==================================================")
    while True:
        user_input = input("\nSiz: ")
        if user_input.lower() == 'exit':
            print("Xayr!")
            break
        
        if not user_input.strip():
            continue
            
        print("\nAgent o'ylamoqda...")
        print("==================================================")
        try:
            agent_answer = run_agent(user_input)
            print(f"\nAgent: {agent_answer}")
        except Exception as e:
            print(f"\nXatolik yuz berdi: {str(e)}")
# Har qanday xatolik yuz berganda, foydalanuvchiga xabar beramiz va davom etamiz. Bu, foydalanuvchi tajribasini yaxshilash uchun muhimdir.



