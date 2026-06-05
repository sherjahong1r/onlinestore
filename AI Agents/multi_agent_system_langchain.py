import os
import requests

# 1. SOZLAMALAR
GROQ_API_KEY = "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"  
MODEL_NAME = "llama-3.3-70b-versatile" # Model nomi, bu yerda siz o'zingiz xohlagan modelni tanlashingiz mumkin. Masalan, "gpt-4" yoki "llama-3.3-70b-versatile". Model nomi API tomonidan qo'llab-quvvatlanadigan modelga mos kelishi kerak.
URL = "https://api.groq.com/openai/v1/chat/completions" # Dastur requests orqali savolni aynan shu manzilga joʻnatadi.
 
headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}", # API kalitini Authorization headeriga qo'shadi. Bu, API serveriga sizning kimligingizni va ruxsatlaringizni tekshirish uchun kerak
    "Content-Type": "application/json" # So'rovning tanasida JSON formatida ma'lumot yuborilishini bildiradi. Bu, API serveriga yuborilayotgan ma'lumotning formatini ko'rsatadi va serverga to'g'ri tarzda ma'lumotni qabul qilish imkonini beradi.
}

# 2. MULTI-AGENT TIZIMI
def run_multi_agent_system(user_question: str):
    # 1-QADAM: ROUTER MANTIQLARI (Xatolikni oldini olish uchun kuchaytirildi)
    router_prompt = (
        "Foydalanuvchi savolini tahlil qil va faqat quyidagi 3 ta so'zdan birini qaytar:\n"
        "- Agar savol ob-havo, iqlim, tumanlar yoki harorat haqida bo'lsa: 'weather'\n"
        "- Agar savol kod yozish, algoritm yoki dasturlash haqida bo'lsa: 'coding'\n"
        "- Qolgan barcha umumiy savollar, salomlashishlar va suhbatlar uchun: 'chat'\n\n"
        "DIQQAT: Faqat bitta so'z yoz! Gap tuzma, nuqta qo'yma. Faqat 'weather', 'coding' yoki 'chat' deb javob ber."
    )
    
    data_router = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": router_prompt},
            {"role": "user", "content": user_question}
        ],
        "temperature": 0
    }
    
    try:
        response = requests.post(URL, json=data_router, headers=headers)
        res_json = response.json()
        
        if 'error' in res_json:
            print(f"\n❌ API Xatoligi: {res_json['error']['message']}")
            return

        # Model ortiqcha gap yozsa ham, faqat kerakli kalit so'zni qidirib olamiz
        raw_content = res_json['choices'][0]['message']['content'].lower()
        
        if "weather" in raw_content:
            selected_agent = "weather"
        elif "coding" in raw_content:
            selected_agent = "coding"
        else:
            selected_agent = "chat"
            
    except Exception as e:
        print(f"\n❌ Routerda kutilmagan chalkashlik: {e}")
        return

    print(f"\n[ROUTER DETECTED]: Savol '{selected_agent.upper()}' agentiga yo'naltirildi.")

    # 2-QADAM: AGENT PROMPTLARI
    if selected_agent == "weather":
        agent_instruction = "Siz geografiya va ob-havo mutaxassisiz. Foydalanuvchining hudud (masalan Zomin yoki boshqa joy) haqidagi iqlimiy/ob-havo savoliga o'zbek tilida ilmiy va tushunarli javob bering."
    elif selected_agent == "coding":
        agent_instruction = "Siz tajribali dasturchisiz. So'ralgan kodni chiroyli yozib, o'zbek tilida tushuntiring."
    else:
        agent_instruction = "Siz aqlli yordamchisiz. Foydalanuvchining umumiy savoliga o'zbekcha chiroyli javob qaytaring."

    # 3-QADAM: YAKUNIY JAVOB
    data_agent = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": agent_instruction},
            {"role": "user", "content": user_question}
        ],
        "temperature": 0.5
    }

    try:
        response_final = requests.post(URL, json=data_agent, headers=headers)
        final_text = response_final.json()['choices'][0]['message']['content']
        print(f"\n🤖 AI Agent Javobi:\n{final_text}\n")
    except Exception as e:
        print(f"❌ Javob olishda xatolik: {e}")

# 3. JONLI CHAT (SAVOLNI SHU YERDA TERMINALDA KIRITASIZ)
if __name__ == "__main__":
    print("Multi-Agent tizimi ishga tushdi!")
    print("Dasturni to'xtatish uchun 'exit' deb yozing.\n")
    
    while True:
        # Savolni dastur kodiga emas, to'g'ridan-to'g'ri terminalga yozasiz!
        savol = input("Savolingizni kiriting: ")
        if savol.lower() == 'exit':
            print("Dastur tugatildi.")
            break
        if savol.strip() == "":
            continue
            
        run_multi_agent_system(savol)
        print("-" * 50) # Har bir savol-javobdan keyin chiziq bilan ajratib turiladi












