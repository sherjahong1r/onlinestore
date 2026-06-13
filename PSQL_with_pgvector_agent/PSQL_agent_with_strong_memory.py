import os
import time
from typing import Annotated, TypedDict
from dotenv import load_dotenv # .env faylidan bazaga ulanish paroli va URL manzilini xavfsiz o'qib olish.
# # LangChain va LangGraph komponentlari

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres import PostgresSaver # Agentning "xotira"si. Suhbat tarixini PostgreSQL-da saqlaydi.
# Biz yozgan baza modullari
from PSQL_db import get_db_schema, execute_sql

load_dotenv()
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
DB_URL = os.getenv("DB_URL")

class AgentState(TypedDict): # Bu agentning "xotira kartasi". U har bir bosqichda xabarlar saqlab turadi
    messages: Annotated[list, add_messages]
    sql_query: str
    db_result: str
    source: str
    table_used: str

# Agent qadamlari (Nodes) router_node (Qaror qabul qiluvchi)
def router_node(state: AgentState):
    question = state["messages"][-1].content
    prompt = f"Savol: {question}. Bu savol bazadan malumot olishni talab qiladimi? Javobni faqat 'SQL' yoki 'LLM' deb qaytaring."
    decision = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return {"source": decision}
# Sql yoki llm ligi aniqlanadi

# sql_generator_node (SQL yaratuvchi)
def sql_generator_node(state: AgentState):
    if state.get("source") != "SQL":
        return {"sql_query": "", "table_used": "N/A"}
    
    question = state["messages"][-1].content
    db_schema = get_db_schema()

#Bu yerda siz AIga 26 ta jadvalingizni va qoidalarini promp sifatida berasiz. AI bazani tushunib, [SQL]: SELECT... ko'rinishida kod yozib beradi.
    system_prompt = f""" 
Sen PostgreSQL mutaxassisisiz. 
Faqat quyidagi sxemadagi jadvallardan SQL yoz:
{db_schema}

QATTIQ QOIDALAR:
- FAQAT yuqoridagi 26 ta jadvaldan foydalanish
- information_schema, pg_catalog kabi tizim jadvallarini ISHLATMA
- Jadval nomini [JADVAL] ga yoz
- SQL ni [SQL] ga yoz

FORMAT (faqat shunday):
[JADVAL]: jadval_nomi
[SQL]: SELECT ...

Savol: {question}
"""
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    response = llm.invoke(messages).content 
# SystemMessage: Bu AIga beriladigan "Yo'riqnoma" (Instructions).
# HumanMessage: Bu Foydalanuvchining savoli. AI aynan shu savolga javob berishi kerak.
# Siz ularni bitta messages ro'yxatiga joylab, AIga "Mana sening qoidalaring (SystemMessage) va mana sening savoling (HumanMessage)" deb beryapsiz.
# llm.invoke(messages): Bu yerda siz AI modeli (Groq orqali Llama-3.3) bilan "gaplashyapsiz". Siz unga yuqoridagi messages ro'yxatini yuborasiz. AI barcha qoidalaringizni (SystemMessage) o'qib chiqadi va savolga (HumanMessage) javob qidiradi.
# .content: AI javob berganida, u o'zi bilan birga ko'plab texnik ma'lumotlarni (tokenlar soni, vaqt va h.k.) ham qaytaradi. .content xossasi faqat bizga kerak bo'lgan matnli javobni (ya'ni AI yozgan SQL kodni) ajratib oladi.

    # [JADVAL] ajratish
    table_name = "Noaniq"
    if "[JADVAL]:" in response:
        table_name = response.split("[JADVAL]:")[1].split("\n")[0].strip()
    
    # [SQL] ajratish
    clean_sql = ""
    if "[SQL]:" in response:
        clean_sql = response.split("[SQL]:")[-1].strip().replace("```sql", "").replace("```", "").strip()
    
    return {"sql_query": clean_sql, "table_used": table_name}

# sql_executor_node (SQL bajaruvchi)
def sql_executor_node(state: AgentState):
    if state.get("source") != "SQL":
        return {}
    result = execute_sql(state["sql_query"])
    return {"db_result": str(result)}
# AI yozgan SQL'ni sizning PSQL_db.py faylingizdagi funksiyaga yuboradi va bazadan natijani olib keladi.

# answer_generator_node (Javob beruvchi)
def answer_generator_node(state: AgentState):
    start_time = time.time()
    question = state["messages"][-1].content
    db_result = state.get("db_result", "Natija yoq")
    history = state["messages"][-10:]
    
    prompt = f"""
    Suhbat tarixi: {history}
    Baza natijasi: {db_result}
    Savol: {question}
    Javobni qisqa va tushunarli ozbekcha bering.
    """
    answer = llm.invoke([HumanMessage(content=prompt)]).content
    duration = round(time.time() - start_time, 2)
    
    table = state.get("table_used", "Noaniq")
    if state.get("source") == "SQL":
        manba = f"PostgreSQL bazasi → '{table}' jadvali"
    else:
        manba = "Llama 3.3 (Groq) suniy intellekti"
    
    final_text = (
        f"{answer}\n"
        f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n"
        f"✅ Manba: {manba}\n"
        f"⏱  Vaqt: {duration} soniya\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    return {"messages": [AIMessage(content=final_text)]}

# Graf qurish
workflow = StateGraph(AgentState)
# Bu yerda biz StateGraph nomli klass yordamida "ishchi graf" (workflow) yaratamiz. AgentState esa agent har bir qadamda o'zi bilan olib yuradigan "xotira kartasi" (xabarlar, SQL so'rov, baza natijasi va h.k.).
# Tugunlar — bu agent bajaradigan ish qadamlari.
workflow.add_node("router", router_node) # "router": Savolni tahlil qilib, SQL kerakmi yoki yo'qmi deb qaror qabul qiladi.
workflow.add_node("sql_gen", sql_generator_node) # "sql_gen": Bazangiz sxemasi asosida SQL kodini yaratadi.
workflow.add_node("executor", sql_executor_node) # "executor": Yaratilgan SQL kodni bazaga yuboradi.
workflow.add_node("answer", answer_generator_node) # "answer": Baza natijasini olib, uni chiroyli javobga aylantiradi.
# Yo'nalishlarni (Edges) belgilash Yo'nalishlar agent bir qadamdan ikkinchisiga qanday o'tishini belgila
workflow.add_edge(START, "router")
workflow.add_edge("router", "sql_gen")
workflow.add_edge("sql_gen", "executor")
workflow.add_edge("executor", "answer")
workflow.add_edge("answer", END)

# PostgreSQL xotira ulash
with PostgresSaver.from_conn_string(DB_URL) as memory:
    memory.setup()  # xotira jadvallarini avtomatik yaratadi
    graph = workflow.compile(checkpointer=memory)
# Bu har bir foydalanuvchining thread_idsi bo'yicha suhbatni bazada saqlaydi. Siz "exit" deb chiqib ketsangiz ham, ertaga kelib "kecha nima deb edim?" desangiz, agent eslaydi.

    if __name__ == "__main__":
        # Har bir foydalanuvchi uchun alohida thread_id
        print('\n')
        thread_id = input("Foydalanuvchi ismini kiriting (masalan: user_1): ")
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"\nAGENT ISHGA TUSHDI! (Men sizga yordam beraman: {thread_id})")
        print("Chiqish uchun 'exit' deb yozing\n")
        
        while True:
            user_input = input("Savolingizni kiriting >>>>>> ")
            if user_input.lower() == 'exit':
                break
            for output in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config  # xotira shu yerda ishlaydi
            ):
                for key, value in output.items():
                    if key == "answer":
                        print(f"\nJavob: {value['messages'][-1].content}")






