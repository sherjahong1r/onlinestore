import sys 
import os
# main.py qaysi papkada bo'lsa, o'sha papkani Python qidiruv yo'liga qo'shadi
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI # veb server yasash uchun
from pydantic import BaseModel # kelgan ma'lumot to'g'ri formatda ekanligini tekshiradi
# agent_sql_LangGraph - bu agent kodi joylashgan fayl nomi
from agent_sql_LangGraph import graph 
from langchain_core.messages import HumanMessage # foydalanuvchi savolini agentga tushunadigan formatga o'giradi

app = FastAPI() # FastAPI serveri ishga tushiriladi. Keyingi barcha endpointlar shu app orqali e'lon qilinadi.

# API ga kelayotgan so'rovda faqat user_input maydoni bo'lishi kerakligini belgilaydi. Masalan: {"user_input": "salom"}. Boshqa narsa kelsa — avtomatik xato qaytaradi.
class QueryRequest(BaseModel):
    user_input: str

@app.post("/ask") #  http://localhost:8000/ask manziliga POST so'rov kelganda bu funksiya ishga tushadi
async def ask_agent(request: QueryRequest):
    # Agentni o'zgarishsiz chaqiramiz
    current_state = {"messages": [HumanMessage(content=request.user_input)]}
    result = graph.invoke(current_state)
 
    # Oxirgi javobni qaytarish
    answer = result["messages"][-1].content 
    return {"answer": answer}
# HumanMessage(...) — savolni agent tushunadigan formatga o'giradi
# graph.invoke(...) — agentni ishga tushiradi, u SQL yoki LLM orqali javob tayyorlaydi
# `result["messages"]
