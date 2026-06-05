import subprocess # Python kod ichidan yangi paketlarni o'rnatish uchun subprocess modulini import qilamiz. Bu modul, terminal buyruqlarini Python ichidan bajarishga imkon beradi. Agar kerakli paketlar o'rnatilmagan bo'lsa, bu modul yordamida ularni avtomatik ravishda o'rnatish mumkin.
import sys # sys moduli, Python interpreterining o'ziga xos xususiyatlarini boshqarish uchun ishlatiladi. Bu modul yordamida, o'rnatilmagan paketlar aniqlanganda, foydalanuvchiga xabar berib, kodni qayta ishga tushirish imkonini yaratamiz.
import os

# Kerakli yangi paketni kod ichidan turib avtomatik o'rnatish
try:
    import langchain_groq
except ImportError:
    print("Groq kutubxonasi o'rnatilmoqda...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-groq", "langchain", "langchain-community"])
    print("O'rnatildi! Kodni qayta ishga tushiring.")
    sys.exit()

from langchain_core.tools import tool
from langchain_groq import ChatGroq  # OpenAI o'rniga Groq
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

# ==========================================
# 1. ASBOBLAR (TOOLS)
# ==========================================
search_tool = DuckDuckGoSearchRun()

@tool
def weather_agent_tool(location: str) -> str:
    """Aniq vaqtdagi ob-havo, bugungi harorat va meteorologik ma'lumotlarni internetdan qidirish uchun."""
    print(f"-> [WEATHER AGENT]: '{location}' uchun qidirilmoqda...")
    try:
        live_data = search_tool.invoke(f"current weather in {location} today")
        return f"Jonli ma'lumot: {live_data}"
    except Exception:
        return f"[Baza] Hozirda {location} shahrida havo iliq va ochiq."

@tool
def coding_agent_tool(query: str) -> str:
    """Dasturlash, kod yozish va xatolarni tuzatishga oid so'rovlar bazasi."""
    print(f"-> [CODING AGENT]: Kod yozish mantiqi ishga tushdi...")
    return "Dasturlash bo'yicha so'rov. Kod muhiti va algoritmlar bazasi tayyor."

@tool # LangChain kutubxonasiga tegishli bo'lgan maxsus dekorator (funksiya ustidan qo'yiladigan belgi).
def general_chat_tool(query: str) -> str: 
    """Oddiy suhbat, umumiy savol-javoblar va chat uchun asbob."""
    print(f"-> [CHAT AGENT]: Umumiy chat ishga tushdi...")
    return "Foydalanuvchi bilan erkin muloqot va umumiy ma'lumotlar bazasi."

# API kaliti:
os.environ["GROQ_API_KEY"] = "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"

# Groq'ning eng mukammal va aqlli modelini ishga tushiramiz
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

tools = [weather_agent_tool, coding_agent_tool, general_chat_tool] # asboblar tools ga saqlandi
llm_with_tools = llm.bind_tools(tools) # tools va llm qo'shildi yani llm.bind_tools(tools) funksiyasi, Groq modeliga yuqorida yaratgan asboblarimizни (tools) bog'laydi. Bu, modelga foydalanuvchi savoliga qarab qaysi asbobни ishlatishни o'rgatadi. Masalan, agar foydalanuvchi ob-havo haqida so'rasa, model weather_agent_tool ни chaqiradi; kod yozish haqida so'rasa, coding_agent_tool ни; va umumiy suhbat uchun general_chat_tool ни ishlatadi. Bu multi-agent tizimining asosiy qismi bo'lib, modelga turli vazifalarni bajarish uchun mos asboblarni tanlash imkonни beradi.
# Bu yerda biz Groq modelini yaratib, unga asboblar (tools) ro'yxatini bog'laymiz. Bu, modelga foydalanuvchi savoliga qarab qaysi asbobни ishlatishни o'rgatadi. Masalan, agar foydalanuvchi ob-havo haqida so'rasa, model weather_agent_tool ни chaqiradi; kod yozish haqida so'rasa, coding_agent_tool ни; va umumiy suhbat uchun general_chat_tool ни ishlatadi. Bu multi-agent tizimining asosiy qismi bo'lib, modelga turli vazifalarni bajarish uchun mos asboblarni tanlash imkonни beradi.

# ==========================================
# 3. MULTI-AGENT FUNKSIYASI
# ==========================================
def run_multi_agent_system(user_question: str):
    print(f"\n==========================================")
    print(f"Foydalanuvchi: '{user_question}'")
    print(f"==========================================")
    
    try:
        ai_msg = llm_with_tools.invoke([HumanMessage(content=user_question)])
        # Foydalanuvchi savolini modelga yuboradi va model javobini ai_msg o'zgaruvchisiga saqlaydi. Bu yerda invoke() funksiyasi, modelga savolni yuborish va javob olish uchun ishlatiladi. Model, foydalanuvchi savoliga qarab, kerakli asbobni chaqiradi va natijani qaytaradi.
        
        if ai_msg.tool_calls: # Agar model javobida asbob chaqirilgan bo'lsa, bu shart bajariladi. ai_msg.tool_calls, modelning javobida qaysi asboblar chaqirilganini ko'rsatadi. Agar hech qanday asbob chaqirilmagan bo'lsa, bu ro'yxat bo'sh bo'ladi va if sharti bajarilmaydi.
            for tool_call in ai_msg.tool_calls: # Model javobida chaqirilgan har bir asbob uchun, bu sikl bajariladi. tool_call, model javobida chaqirilgan har bir asbobni ifodalaydi. Bu yerda biz, modelning javobida qaysi asbob chaqirilganini aniqlaymiz va unga mos ravishda kerakli amallarni bajarish uchun ishlatamiz.
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                print(f"[ROUTER DETECTED]: Savol '{tool_name}' asbobiga mos keldi.")
                
                if tool_name == "weather_agent_tool":
                    tool_output = weather_agent_tool.invoke(tool_args)
                elif tool_name == "coding_agent_tool":
                    tool_output = coding_agent_tool.invoke(tool_args)
                elif tool_name == "general_chat_tool":
                    tool_output = general_chat_tool.invoke(tool_args)
                
                final_response = llm.invoke([ # Modelga foydalanuvchi savolini, model javobini va asbob natijasini yuborib, yakuniy chiroyli javob olish uchun ishlatiladi. Bu yerda biz, modelga foydalanuvchi savolini (HumanMessage), modelning o'z javobini (ai_msg) va asbobdan olingan natijani (tool_output) yuboramiz. Model, bu ma'lumotlarni tahlil qilib, foydalanuvchiga o'zbek tilida chiroyli va tushunarli javob qaytaradi.
                    HumanMessage(content=user_question), # Foydalanuvchi savoli, bu yerda modelga yuboriladigan asosiy savolni ifodalaydi. Bu savol, modelga nima haqida javob berish kerakligini bildiradi.
                    ai_msg, # Modelning o'z javobi, bu yerda modelning asbobni chaqirishdan oldingi javobini ifodalaydi. Bu javob, modelga asbobdan olingan natijani qanday ishlatish kerakligini tushunishga yordam beradi.
                    HumanMessage(content=f"Baza natijasi: {tool_output}. Foydalanuvchiga o'zbek tilida yakuniy chiroyli javob yozib ber.")
                ])
                print(f"\n🤖 AI Agent Javobi:\n{final_response.content}")
                
        else:
            print(f"\n🤖 AI Agent Javobi (To'g'ridan-to'g'ri):\n{ai_msg.content}") 
            
    except Exception as e:
        print(f"\n❌ Xatolik yuz berdi: {e}")

# ==========================================
# 4. JONLI INTERFAOL TERMINAL CHAT
# ==========================================
if __name__ == "__main__": # Bu shart, Python faylini to'g'ridan-to'g'ri ishga tushirilganda bajariladigan kod blokini belgilaydi. Agar fayl boshqa modul tomonidan import qilinsa, bu kod bloklari bajarilmaydi. Bu yerda biz, foydalanuvchidan savol qabul qilish va multi-agent tizimini ishga tushirish uchun interfaol terminal chat yaratamiz.
    print("==================================================")
    print("🤖 Multi-Agent LangChain Tizimi Ishga Tushdi!")
    print("Dasturdan chiqish uchun 'exit' deb yozing.")
    print("==================================================\n")
    
    while True:
        # Savolni shu yerda terminalning o'zida jonli qabul qiladi
        user_input = input("Siz: ")
        
        # Chiqish shartini tekshirish
        if user_input.strip().lower() == 'exit':
            print("\n🤖 Tizim to'xtatildi. Salomat bo'ling!")
            break
            
        # Bo'sh joy tashlab Enter bosilsa, o'tkazib yuboradi
        if not user_input.strip():
            continue
            
        # Funksiyani chaqirish
        run_multi_agent_system(user_input) # run_multi_agent_system funksiyasi, foydalanuvchi savolini qabul qiladi va multi-agent tizimini ishga tushiradi. Bu funksiya, foydalanuvchi savoliga qarab, modelga savolni yuboradi, kerakli asbobni chaqiradi va yakuniy javobni chiqaradi.
        print("\n" + "_"*60 + "\n") # Har bir savol-javobdan keyin chiziq bilan ajratib turiladi, bu esa terminalda javoblarni osonroq ko'rish va ajratib olish imkonini beradi.