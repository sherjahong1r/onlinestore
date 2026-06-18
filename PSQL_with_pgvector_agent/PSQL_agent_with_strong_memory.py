"""
PSQL_agent_with_strong_memory.py
---------------------------------
LangGraph asosida ishlaydigan agent:

    1. Foydalanuvchi savol beradi.
    2. Router savolni tahlil qiladi:
         - Agar savol ma'lumotlar bazasidagi narsa haqida bo'lsa (jadval,
           statistika, ro'yxat va h.k.) -> SQL yo'li orqali javob topiladi.
         - Aks holda -> Groq LLM o'zi javob beradi (umumiy bilim, salomlashish,
           xulosa chiqarish va h.k.)
    3. Har bir javobda foydalanuvchiga ko'rsatiladi:
         - Manba (PostgreSQL / Groq LLM)
         - Agar PostgreSQL bo'lsa - qaysi jadval(lar)dan olingani
         - Javobni generatsiya qilishga ketgan vaqt
    4. Agent kuchli xotiraga ega: PostgresSaver checkpointer orqali har bir
       foydalanuvchi (thread_id) uchun butun suhbat tarixi saqlanadi, shuning
       uchun u "yuqorida nima so'ragan edim" kabi savollarga va bir-biriga
       mantiqiy bog'langan ketma-ket savollarga to'g'ri javob bera oladi.
""" 
"""
PSQL_agent_with_strong_memory.py
---------------------------------
LangGraph asosida ishlaydigan agent. Uchta yo'ldan biri orqali javob beradi:

    1. SQL     -> aniq, struktura asosidagi savollar (necha xodim, eng
                  ko'p oylik, ro'yxat va h.k.) - PostgreSQL'dan oddiy SQL
                  orqali javob topiladi.
    2. VECTOR  -> mazmunan/ma'no bo'yicha qidirish kerak bo'lgan savollar
                  (masalan "shunga o'xshash mahsulot", "bunga yaqin sharh"
                  kabi) - pgvector orqali semantik qidiruv qilinadi.
    3. LLM     -> ma'lumotlar bazasiga aloqasi yo'q umumiy savollar -
                  Groq LLM o'zi javob beradi.
"""
import os
import time
from typing import Annotated, TypedDict, List, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer # "Matnni raqamga aylantiruvchi qism" (Embedding) hisoblanadi.

from PSQL_db import get_db_schema, execute_sql, search_vector_db, get_vector_enabled_tables

# ---------------------------------------------------------------------------
# Konfiguratsiya
# ---------------------------------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_URL = os.getenv("DB_URL")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY topilmadi. .env faylida belgilang.")
if not DB_URL:
    raise RuntimeError("DB_URL topilmadi. .env faylida belgilang.")

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0)

# Bazaning sxemasini bir marta yuklab olamiz (keshlab qo'yamiz, har bir
# savolda qayta so'ramaslik uchun). Bazaga yangi jadval qo'shilsa, dasturni
# qayta ishga tushirish kerak.
DB_SCHEMA_CACHE = get_db_schema()
VECTOR_TABLES_CACHE = get_vector_enabled_tables()

print("Embedding modeli yuklanmoqda (Sentence-Transformers)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding modeli tayyor.\n")


def embed_text(text: str):
    """Matnni vectorga aylantiradi (pgvector qidiruvi uchun)."""
    return embedding_model.encode(text).tolist()


# ---------------------------------------------------------------------------
# Agent holati (State)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    route: str            # "SQL" | "VECTOR" | "LLM"
    db_result_text: str   # baza natijasining matn ko'rinishi
    tables_used: List[str]
    start_time: float
    source_label: str     # foydalanuvchiga ko'rsatiladigan manba nomi
# Vazifasi: Bu Agentning "xotira holati". Agentning har bir qadami shu AgentState ichida saqlanadi.
# Suhbat tarixi, qaysi yo‘ldan ketgani va qaysi jadvalni ishlatgani shu yerda turadi.

# ---------------------------------------------------------------------------
# Node 1: Router - savolni SQL / VECTOR / LLM yo'liga yo'naltirish
# ---------------------------------------------------------------------------

vector_tables_description = "\n".join(
    f"- {table} (matn ustunlari: {', '.join(info['text_columns']) or 'yo`q'})"
    for table, info in VECTOR_TABLES_CACHE.items()
) or "(hozircha vector ustuniga ega jadval yo'q)"

ROUTER_PROMPT = """Sen so'rovlarni yo'naltiruvchi yordamchisan.

Ma'lumotlar bazasi sxemasi:
{schema}

Semantik (ma'no bo'yicha) qidiruv mumkin bo'lgan jadvallar (pgvector bilan):
{vector_tables}

Foydalanuvchi savolini o'qib, FAQAT bitta so'z bilan javob ber:
- "SQL"    -> agar savolga aniq, struktura asosidagi SQL so'rov bilan javob
              berish mumkin bo'lsa (son, ro'yxat, taqqoslash, filtr, aniq qiymat).
- "VECTOR" -> agar savol mazmunan o'xshashlik, "shunga o'xshash", "qanday
              fikrlar bor", "bog'liq narsalarni top" kabi semantik qidiruv
              talab qilsa, VA mos jadval yuqorida ro'yxatda bo'lsa.
- "LLM"    -> agar savol ma'lumotlar bazasiga umuman aloqasi bo'lmagan
              umumiy bilim, salomlashish yoki suhbat haqida bo'lsa.

Faqat "SQL", "VECTOR" yoki "LLM" so'zini yoz, boshqa hech narsa yozma.

Suhbat tarixi:
{history}

Foydalanuvchining oxirgi savoli: {question}
"""

# Vazifasi: Bu eng muhim qadam. Router foydalanuvchi savolini o‘qiydi va LLM (Groq) orqali qaror qabul qiladi:
def router_node(state: AgentState):
    start = time.time()
    question = state["messages"][-1].content

    history_messages = state["messages"][:-1][-6:]
    history_text = "\n".join(
        f"{'Foydalanuvchi' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in history_messages
    ) or "(hali suhbat tarixi yo'q)"

    prompt = ROUTER_PROMPT.format(
        schema=DB_SCHEMA_CACHE,
        vector_tables=vector_tables_description,
        history=history_text,
        question=question,
    )

    decision = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()

    if "VECTOR" in decision:
        route = "VECTOR"
    elif "SQL" in decision:
        route = "SQL"
    else:
        route = "LLM"

    return {"route": route, "start_time": start}


# ---------------------------------------------------------------------------
# Node 2a: SQL yo'li - bazadan ma'lumot olish
# ---------------------------------------------------------------------------

SQL_GENERATION_PROMPT = """Sen PostgreSQL bo'yicha mutaxassissan.
Quyidagi ma'lumotlar bazasi sxemasidan foydalanib, foydalanuvchi savoliga
javob beradigan TO'G'RI va XAVFSIZ SQL so'rovini yoz.

QOIDALAR:
- Faqat SELECT so'rovi yoz (INSERT, UPDATE, DELETE, DROP taqiqlangan).
- Faqat sxemada mavjud jadval va ustun nomlaridan foydalan.
- Natijada faqat toza SQL kodi bo'lsin, izoh yoki ```sql belgilarisiz.
- Agar kerak bo'lsa, suhbat tarixidagi kontekstdan foydalanib oldingi
  savol bilan bog'liq so'rovni tuzishing mumkin.

Sxema:
{schema}

Suhbat tarixi:
{history}

Foydalanuvchi savoli: {question}

SQL:"""

# Mantiqi: LLMga butun sxemani berib, undan SQL yozishni so‘raydi. Agar LLM "yomon" SQL yozsa, 
# baza xato beradi, agent esa o‘sha xatoni chiroyli qilib foydalanuvchiga yetkazadi.
def sql_node(state: AgentState):
    question = state["messages"][-1].content
    history_messages = state["messages"][:-1][-6:]
    history_text = "\n".join(
        f"{'Foydalanuvchi' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in history_messages
    ) or "(hali suhbat tarixi yo'q)"

    prompt = SQL_GENERATION_PROMPT.format(
        schema=DB_SCHEMA_CACHE, history=history_text, question=question
    )

    raw_sql = llm.invoke([HumanMessage(content=prompt)]).content
    sql = raw_sql.replace("```sql", "").replace("```", "").strip()

    result = execute_sql(sql)

    if result["success"]:
        rows = result["rows"]
        if not rows:
            result_text = f"SQL so'rov bajarildi, lekin natija topilmadi.\nSQL: {sql}"
        else:
            preview = rows[:50]
            result_text = (
                f"SQL: {sql}\n"
                f"Natija ({len(rows)} qator, dastlabki {len(preview)} tasi ko'rsatilmoqda):\n{preview}"
            )
    else:
        result_text = f"SQL xato berdi.\nSQL: {sql}\nXato: {result['error']}"

    return {
        "db_result_text": result_text,
        "tables_used": result["tables_used"],
        "source_label": "PostgreSQL",
    }


# ---------------------------------------------------------------------------
# Node 2b: VECTOR yo'li - pgvector orqali semantik qidiruv
# ---------------------------------------------------------------------------

# Mantiqi: Bu qism SQL yozmaydi, balki pgvector orqali "ma'no" bo‘yicha qidiradi. Masalan,
# "kompaniya madaniyati haqida yoz" desa, bazadagi matnlarni solishtirib mosini topadi.
def vector_node(state: AgentState):
    question = state["messages"][-1].content

    result = search_vector_db(question, embed_fn=embed_text, top_k=5)

    if result["success"]:
        rows = result["rows"]
        if not rows:
            result_text = "Semantik qidiruv natija bermadi."
        else:
            result_text = (
                f"Eng o'xshash {len(rows)} natija topildi "
                f"('{result['table_used']}' jadvalidan):\n{rows}"
            )
        tables_used = [result["table_used"]] if result["table_used"] else []
    else:
        result_text = f"Vector qidiruvda xato: {result['error']}"
        tables_used = [result["table_used"]] if result["table_used"] else []

    return {
        "db_result_text": result_text,
        "tables_used": tables_used,
        "source_label": "pgvector",
    }


# ---------------------------------------------------------------------------
# Node 2c: LLM yo'li - Groq o'zi javob beradi (bazaga murojaat qilinmaydi)
# ---------------------------------------------------------------------------

def llm_only_node(state: AgentState):
    return {
        "db_result_text": "",
        "tables_used": [],
        "source_label": "Groq LLM",
    }


# ---------------------------------------------------------------------------
# Node 3: Yakuniy javobni shakllantirish
# ---------------------------------------------------------------------------

FINAL_ANSWER_PROMPT = """Sen foydalanuvchiga yordam beruvchi do'stona agentsan.
Quyida butun suhbat tarixi berilgan - undan foydalanib, savollar orasidagi
mantiqiy bog'liqlikni hisobga ol (masalan, "u" yoki "shuni" kabi so'zlar
oldingi xabarga ishora qilishi mumkin, yoki foydalanuvchi avval nima
so'raganini so'rashi mumkin).

{db_context}

Foydalanuvchining oxirgi savoli: {question}

Javobni o'zbek tilida, tushunarli, qisqa va aniq tarzda yoz. Agar baza
natijasi berilgan bo'lsa, shu natijaga asoslanib javob ber. Agar baza
natijasida xato bo'lsa, buni foydalanuvchiga muloyim tarzda tushuntir."""

# Mantiqi: Bu yerda foydalanuvchiga ko‘rinadigan yakuniy javob tayyorlanadi. Agent bu yerda 
# juda aqlli: u oldingi suhbatni eslaydi (conversation_history), shuning uchun "U qachon ochilgan?" 
# desa, "U" kimligini tushunadi.
def answer_node(state: AgentState):
    duration = time.time() - state.get("start_time", time.time())
    question = state["messages"][-1].content
    db_result_text = state.get("db_result_text", "")

    if state["route"] in ("SQL", "VECTOR"):
        db_context = f"Ma'lumotlar bazasidan olingan natija:\n{db_result_text}"
    else:
        db_context = "(Bu savol uchun ma'lumotlar bazasiga murojaat qilinmadi, umumiy bilim asosida javob ber.)"

    conversation_history = state["messages"][:-1]

    messages_for_llm = [
        SystemMessage(
            content=(
                "Sen foydalanuvchi bilan uzluksiz suhbat olib boruvchi yordamchisan. "
                "Oldingi xabarlarni eslab qol va savollar orasidagi bog'liqlikni tushun."
            )
        )
    ]
    messages_for_llm.extend(conversation_history)
    messages_for_llm.append(
        HumanMessage(
            content=FINAL_ANSWER_PROMPT.format(db_context=db_context, question=question)
        )
    )

    answer_text = llm.invoke(messages_for_llm).content

    source_label = state.get("source_label", "Noma'lum")
    tables_used = state.get("tables_used", [])

    footer_lines = [
      "\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ",
        f"📍 Manba: {source_label}",
    ]
    if source_label in ("PostgreSQL", "pgvector") and tables_used:
        footer_lines.append(f"📋 Jadval: {', '.join(tables_used)}")
    footer_lines.append(f"⏱ Javob vaqti: {duration:.2f} soniya")
    footer_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    final_text = answer_text + "\n" + "\n".join(footer_lines)

    return {"messages": [AIMessage(content=final_text)]}


# ---------------------------------------------------------------------------
# Grafni qurish
# ---------------------------------------------------------------------------

def route_decision(state: AgentState) -> Literal["sql_node", "vector_node", "llm_only_node"]:
    if state["route"] == "SQL":
        return "sql_node"
    elif state["route"] == "VECTOR":
        return "vector_node"
    return "llm_only_node"

# Vazifasi: LangGraph yordamida yo‘llarni belgilaydi. checkpointer (PostgresSaver) esa suhbatni 
# bazaga yozib boradi, shunda siz dasturni o‘chirib yonsangiz ham, tarix saqlanib qoladi.
def build_graph(checkpointer):
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("sql_node", sql_node)
    workflow.add_node("vector_node", vector_node)
    workflow.add_node("llm_only_node", llm_only_node)
    workflow.add_node("answer_node", answer_node)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "sql_node": "sql_node",
            "vector_node": "vector_node",
            "llm_only_node": "llm_only_node",
        },
    )
    workflow.add_edge("sql_node", "answer_node")
    workflow.add_edge("vector_node", "answer_node")
    workflow.add_edge("llm_only_node", "answer_node")
    workflow.add_edge("answer_node", END)

    return workflow.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Dasturni ishga tushirish
# ---------------------------------------------------------------------------

def main():
    with PostgresSaver.from_conn_string(DB_URL) as checkpointer:
        checkpointer.setup()
        graph = build_graph(checkpointer)

        thread_id = input("Foydalanuvchi ismini kiriting (masalan: user_1): ").strip() or "default_user"
        config = {"configurable": {"thread_id": thread_id}}

        print(f"\nAGENT ISHGA TUSHDI! (Sizga qanday yordam kerak: {thread_id})")
        print("Chiqish uchun 'exit' deb yozing.\n")

        while True:
            user_input = input("Savolingizni kiriting >>>>> ").strip()
            if user_input.lower() in ("exit", "quit", "chiqish"):
                print("Xayr!")
                break
            if not user_input:
                continue

            try:
                result_state = graph.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )
                last_message = result_state["messages"][-1]
                print(f"\n{last_message.content}\n")
            except Exception as e:
                print(f"\n[XATO] Javob berishda muammo yuz berdi: {e}\n")


if __name__ == "__main__":
    main()
