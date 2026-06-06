import os # LCEL uchun operatsion tizim bilan ishlash uchun kerak bo'lgan kutubxona
import json 
from langchain_community.vectorstores import Chroma # LCEL uchun Chroma vektor bazasi modeli
from langchain_huggingface import HuggingFaceEmbeddings # LCEL uchun HuggingFace embedding modeli
from langchain_core.documents import Document # LCEL uchun hujjat modeli
from langchain_core.prompts import ChatPromptTemplate # LCEL uchun chat prompt modeli
from langchain_core.runnables import RunnablePassthrough # LCEL uchun oddiy o'tkazuvchi (passthrough) modeli
from langchain_core.output_parsers import StrOutputParser # LCEL uchun matnli javoblarni to'g'ri formatlash uchun parser
from langchain_groq import ChatGroq 

# 1. SOZLAMALAR (Groq tekin kalitini o'rnatish)
# Kalitni olish juda oson: console.groq.com saytiga Google akkauntingiz bilan kirib olasiz
os.environ["GROQ_API_KEY"] = "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"

DB_DIR = "./constitution_chroma_db" # Vektor bazasini saqlash uchun papka nomi. Agar bu papka mavjud bo'lmasa, kod uni yaratadi va unda vektor bazasini saqlaydi. Agar papka allaqachon mavjud bo'lsa, undagi baza yuklanadi. Bu papka ichida Chroma modeli tomonidan yaratilgan vektorlar va indekslar saqlanadi, shunda keyingi safar kod ishga tushirilganda tezda ularga murojaat qilish mumkin bo'ladi.
JSON_FILE = "constitution_uz.json" # Konstitutsiya matnini o'z ichiga olgan JSON fayl nomi. Bu fayl quyidagi formatda bo'lishi kerak: title va text kalitlari bilan, har bir element konstitutsiyaning bir bo'limi yoki moddasi haqida ma'lumot saqlaydi. Kod bu faylni o'qib, undagi matnlarni vektor bazasiga joylashtiradi, shunda savollar berilganda tezda kerakli ma'lumotlarni qidirib topish mumkin bo'ladi.


# 2. EMBEDDING MODELNI YUKLASH
print("Embedding model yuklanmoqda (intfloat/multilingual-e5-base)...") 
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
# Internetdan ko'p tillarni (jumladan o'zbek tilini ham) juda yaxshi tushunadigan multilingual-e5-base modelini kompyuter xotirasiga yuklaydi. U matnlarni ma'nosiga qarab vektorga aylantiradi.

# 3. VEKTOR BAZANI YUKLASH YOKI YARATISH
if not os.path.exists(DB_DIR): 
    print("Vektor baza topilmadi. Yangi baza yaratilmoqda...") # Agar kodingiz turgan joyda constitution_chroma_db degan papka bo'lmasa, yangidan yaratish bosqichini boshlaydi.
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f: # constitution_uz.json faylini o'qib, undagi har bir elementni Document formatiga o'tkazadi. Har bir elementning "text" qismi page_content sifatida, "title" qismi esa metadata ichida source sifatida saqlanadi.
            data = json.load(f) # JSON faylni o'qib, Python ro'yxatiga aylantiradi. Har bir element bu ro'yxatda lug'at (dictionary) shaklida bo'ladi, unda "title" va "text" kalitlari mavjud.
        documents = [
            Document(page_content=item["text"], metadata={"source": item["title"]}) # JSON fayldagi har bir element uchun Document obyektini yaratadi. page_content qismiga "text" qiymati, metadata ichida esa "title" qiymati saqlanadi. Bu hujjatlar keyinchalik vektor bazasiga joylashtiriladi.
            for item in data
        ]
        vector_store = Chroma.from_documents( # Document obyektlarini va embedding modelini olib, yangi vektor bazasini yaratadi. Bu baza keyinchalik diskda DB_DIR papkasida saqlanadi.
            documents=documents,
            embedding=embeddings,
            persist_directory=DB_DIR
        ) # Tayyor hujjatlarni embedding modeli orqali vektorga o'giradi va kompyuteringizda Chroma bazasi ko'rinishida qattiq diskka (DB_DIR) saqlab qo'yadi.
        print("Vektor baza muvaffaqiyatli yaratildi!")
    except FileNotFoundError:
        print(f"Xato: Kompyuterda '{JSON_FILE}' fayli topilmadi!")
        exit() # Agar constitution_uz.json fayli mavjud bo'lmasa, foydalanuvchiga xato haqida xabar beradi va dasturdan chiqadi.
else:
    print("Mavjud vektor baza yuklanmoqda...")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings) # Agar DB_DIR papkasi allaqachon mavjud bo'lsa, undagi saqlangan vektor bazasini yuklab oladi. Bu baza embedding modeli yordamida yaratilgan va keyinchalik savollarni tezkor javoblash uchun ishlatiladi.

# 4. GROQ ORQALI LLAMA3 MODELINI TEKIN ULASH
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3) # Groq platformasida mavjud bo'lgan Llama 3.1 modelini chaqiradi. Bu model juda kuchli va ko'p tillarni tushunishga qodir, shuning uchun u O'zbekiston Respublikasi Konstitutsiyasi bo'yicha savollarga aniq va ma'noli javoblar berishi mumkin. temperature=0.3 parametri modelning javoblarini biroz barqaror va aniqroq qilish uchun sozlangan, bu esa huquqiy savollar uchun muhimdir.

system_prompt = (
    "Siz O'zbekiston Republicasi Konstitutsiyasi bo'yicha ixtisoslashgan aqlli huquqshunos yordamchisiz.\n"
    "Faqat quyida berilgan kontekst (moddalar) ma'lumotlaridan foydalanib savolga javob bering.\n"
    "Javobingiz qat'iy, tushunarli va huquqiy o'zbek tilida bo'lsin.\n\n"
    "Kontekst:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([ 
    ("system", system_prompt), # Tizim yo'riqnomasi (system_prompt) ni chat promptining birinchi qismi sifatida belgilaydi. Bu qismda agentning vazifasi va javob berish uslubi haqida aniq ko'rsatmalar berilgan.
    ("human", "{input}"), 
]) # Tizim yo'riqnomasi (system_prompt) va foydalanuvchi bergan jonli savol ({input})ni birlashtirib, yakuniy xabar paketini tayyorlaydi.

# --- RAG ZANJIRINI TO'G'RI QURISH ---
# Obyektni retriever formatiga xavfsiz o'tkazib olamiz
retriever = vector_store.as_retriever(search_kwargs={"k": 2}) # Vektor bazasidan eng yaqin 2 ta hujjatni qidirib olish uchun retriever obyektini yaratadi. search_kwargs={"k": 2} parametri har bir savol uchun 2 ta eng mos hujjatni qaytarishni belgilaydi.

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# Bazadan topilgan 2 ta hujjat obyektini (Document) olib, ularning ichidagi matnlarni o'rtasiga qator tashlab, bitta yaxlit matn holatiga keltiruvchi yordamchi funksiya.

# Mukammal va xatoliksiz zanjir (LCEL)
rag_chain = ( # RAG (Retrieval-Augmented Generation) zanjiri quyidagi bosqichlardan iborat:
    {"context": retriever | format_docs, "input": RunnablePassthrough()} # Savol kelganda, retriever yordamida eng yaqin 2 ta hujjatni qidirib olib, ularni format_docs funksiyasi orqali bitta matn holatiga keltiradi. Foydalanuvchi savoli esa o'zgarishsiz (passthrough) tarzda saqlanadi.
    | prompt # Tizim yo'riqnomasi va foydalanuvchi savolini birlashtirib, yakuniy xabar paketini tayyorlaydi. 
    | llm  # Tayyorlangan xabar paketini Llama 3.1 modeli orqali qayta ishlaydi va javobni generatsiya qiladi. Modelga kontekst va savol birga beriladi, shunda u aniq va ma'noli javob yaratishi mumkin.
    | StrOutputParser() # Modeldan olingan javobni matn formatida to'g'ri chiqarish uchun parser. Bu bosqichda modelning javobi oddiy matn sifatida olinadi va foydalanuvchiga ko'rsatilishga tayyorlanadi.
)

# 5. AGENTNI ISHGA TUSHIRISH
print("\n--- RAG Agent Tayyor! (O'zbekiston Republikasi Konstitutsiyasiga asoslangan) ---")
while True:
    query = input("\nSavolingizni bering (Chiqish uchun 'exit' deb yozing): ")
    if query.lower() == 'exit': 
        break
    if query.strip() == "": 
        continue
        
    print("Agent qidirmoqda va o'ylamoqda...")
    try:
        response = rag_chain.invoke(query)
        print("\n[JAVOB]:", response)
    except Exception as e:
        print("\nXatolik yuz berdi:", e)




















