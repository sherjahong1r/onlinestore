"""
api.py
------
PSQL_agent_with_strong_memory.py ichidagi agentni HTTP API sifatida ochib
beradi. Bu fayl alohida ishga tushiriladi (PSQL_agent_with_strong_memory.py
ning o'zi emas), chunki u terminal input() o'rniga HTTP so'rovlarni qabul
qiladi.

Ishga tushirish:
    python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000

Sinab ko'rish (boshqa terminalda):
    curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d "{\"thread_id\": \"user_1\", \"message\": \"nechta mahsulot bor\"}"

Yoki brauzerda avtomatik generatsiya qilingan hujjatni ko'rish uchun:
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

from PSQL_agent_with_strong_memory import build_graph, DB_URL

# ---------------------------------------------------------------------------
# Checkpointer va grafni dastur boshlanganda BIR MARTA ochib, butun dastur
# umri davomida ochiq saqlaymiz (har bir so'rovda qayta ulanish sekin va
# resurs isrof qiladi).
# ---------------------------------------------------------------------------

checkpointer_cm = None
graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global checkpointer_cm, graph
    checkpointer_cm = PostgresSaver.from_conn_string(DB_URL)
    checkpointer = checkpointer_cm.__enter__()
    checkpointer.setup()
    graph = build_graph(checkpointer)
    print("Agent va baza ulanishi tayyor. API ishga tushdi.")

    yield  # <-- dastur shu yerda ishlaydi (so'rovlarni qabul qiladi)

    checkpointer_cm.__exit__(None, None, None)
    print("Baza ulanishi yopildi.")


app = FastAPI(
    title="PostgreSQL + pgvector + Groq Agent API",
    description="SQL, semantik (pgvector) va umumiy savollarga javob beruvchi agent",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - boshqa domendan (masalan frontend saytdan) so'rov yuborilishi uchun.
# Sinov maqsadida hammaga ochiq qoldirilgan, productionda aniq domenlarni
# ko'rsatish tavsiya etiladi.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# So'rov / javob formatlari
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    thread_id: str   # foydalanuvchini ajratish uchun (xotira shu bo'yicha saqlanadi)
    message: str      # foydalanuvchi savoli


class ChatResponse(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# Endpoint'lar
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Agent API ishlayapti. /docs orqali sinab ko'ring."}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message bo'sh bo'lishi mumkin emas.")

    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        result_state = graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )
        answer = result_state["messages"][-1].content
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent xatosi: {e}")





