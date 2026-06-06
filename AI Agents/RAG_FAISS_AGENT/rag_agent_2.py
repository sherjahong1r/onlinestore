import os
from langchain_community.document_loaders import PyPDFLoader # PDF fayllarni o'qish uchun maxsus loader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Matnni belgilangan o'lchamda va uzilishlarni oldini olish uchun ustma-ust tushish bilan bo'lish
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # FAISS vektor bazasini yaratish va boshqarish uchun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough # Oddiy o'tkazuvchi (passthrough) modeli, foydalanuvchi savolini o'zgartirmasdan saqlash uchun
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# 1. API KALIT VA SOZLAMALAR
os.environ["GROQ_API_KEY"] = "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" # Groq API kaliti. Bu kalit Groq platformasiga ulanish va Llama 3.1 modelidan foydalanish uchun zarur. Kalitni maxfiy saqlang va hech qachon ommaga oshkor qilmang, chunki bu sizning hisobingizga ruxsatsiz kirishga olib kelishi mumkin.

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # Kod faylining joylashgan papkasini aniqlaydi. Bu yerda PDF fayl va FAISS bazasini saqlash uchun ishlatiladi. Bu usul kodni boshqa joyga ko'chirish yoki boshqalar bilan bo'lishish osonroq bo'lishini ta'minlaydi, chunki yo'llar mutloq (absolute) tarzda belgilanadi.

# PDF fayl va FAISS bazasi yo'lini o'sha papkaga mutloq (absolute) qilib bog'laymiz
PDF_FILE = os.path.join(CURRENT_DIR, "ajab_dunyo.pdf") # Bu yerda "ajab_dunyo.pdf" nomli PDF faylni CURRENT_DIR papkasiga joylashtiring. Kod shu faylni o'qib, matnni chunklarga bo'lib, embedding modelidan o'tkazib, FAISS bazasini yaratadi. Agar bu fayl mavjud bo'lmasa, kod xatolik beradi, shuning uchun fayl nomini va joylashuvini tekshiring.
FAISS_DIR = os.path.join(CURRENT_DIR, "ajab_dunyo_faiss") # FAISS bazasini saqlash uchun "ajab_dunyo_faiss" nomli papka. Agar bu papka mavjud bo'lmasa, kod yangi FAISS bazasini yaratadi va uni shu papkaga saqlaydi. Agar papka allaqachon mavjud bo'lsa, undagi saqlangan FAISS bazasini yuklab oladi. Bu usul kodni boshqa joyga ko'chirish yoki boshqalar bilan bo'lishish osonroq bo'lishini ta'minlaydi, chunki yo'llar mutloq (absolute) tarzda belgilanadi.

# 2. HUGGING FACE EMBEDDING MODELINI YUKLASH
print("Hugging Face embedding modeli yuklanmoqda (multilingual-e5-base)...")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base") # Bu model ko'p tillarni, jumladan o'zbek tilini ham, juda yaxshi tushunadi. Matnlarni ma'nosiga qarab vektorlarga aylantiradi, bu esa keyinchalik FAISS bazasida tezkor qidiruv uchun zarur.

# 3. PDF KITOBNI CHUNKLARGA BO'LISH VA FAISS BAZASINI YARATISH
if not os.path.exists(FAISS_DIR): # Agar FAISS_DIR papkasi mavjud bo'lmasa, yangi baza yaratish bosqichini boshlaydi. Bu yerda PDF faylni o'qib, matnni belgilangan o'lchamda chunklarga bo'lib, keyinchalik bu chunklarni embedding modelidan o'tkazib, FAISS bazasiga joylashtiradi.  
    print(f"\n[YANGI BAZA]: '{PDF_FILE}' kitobi tizimga o'qitilmoqda...")
    try:
        # PDF faylni o'qiymiz
        loader = PyPDFLoader(PDF_FILE)
        raw_docs = loader.load()
        print(f"Kitob muvaffaqiyatli o'qildi. Jami sahifalar soni: {len(raw_docs)}")
        
        # 330 betlik badiiy asar uchun chunk hajmini 1000 ta belgi qilib olamiz
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,     # Har bir chunkning maksimal uzunligi (belgilar soni)  
            chunk_overlap=200,     # Hikoyalar mazmuni uzilib qolmasligi uchun ustma-ust tushish darchasi
            length_function=len
        )
        chunks = text_splitter.split_documents(raw_docs) # PDF fayldan o'qilgan matnni belgilangan o'lchamda va uzilishlarni oldini olish uchun ustma-ust tushish bilan chunklarga bo'ladi. Har bir chunk badiiy asarning kichik bir qismi bo'lib, bu chunklar keyinchalik embedding modelidan o'tkazilib, FAISS bazasiga joylashtiriladi. Bu usul savollarga javob berishda aniq va ma'noli javoblar olish uchun zarur, chunki badiiy asarlar ko'pincha uzun va murakkab bo'ladi.
        print(f"Matn muvaffaqiyatli chunklarga bo'lindi. Jami chunklar: {len(chunks)}")
        
        # FAISS bazasini yaratamiz va diskka saqlaymiz
        print("Vektorlar yaratilib, FAISS bazasiga muhrlanmoqda... (Biroz vaqt olishi mumkin)")
        vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings) # Chunklarga bo'lingan matnni embedding modeli orqali vektorlarga aylantiradi va FAISS bazasini yaratadi. Bu baza keyinchalik diskka saqlanadi, shunda keyingi safar kod ishga tushirilganda tezda ularga murojaat qilish mumkin bo'ladi.
        vector_store.save_local(FAISS_DIR) # Yaratilgan FAISS bazasini diskka saqlaydi. Bu papka ichida FAISS modeli tomonidan yaratilgan vektorlar va indekslar saqlanadi, shunda keyingi safar kod ishga tushirilganda tezda ularga murojaat qilish mumkin bo'ladi.
        print("FAISS vektor bazasi muvaffaqiyatli pishirildi!")
        
    except FileNotFoundError:
        print(f"Xatolik: Papkada '{PDF_FILE}' nomli fayl topilmadi! Kitob nomini tekshiring.")
        exit()
else:
    print("\n[MAVJUD BAZA]: Avvaldan tayyor bo'lgan FAISS indeksi yuklanmoqda...")
    # Saqlangan FAISS bazasini diskdan qayta xotiraga yuklaymiz
    vector_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True) # Agar FAISS_DIR papkasi allaqachon mavjud bo'lsa, undagi saqlangan FAISS bazasini yuklab oladi. Bu baza embedding modeli yordamida yaratilgan va keyinchalik savollarni tezkor javoblash uchun ishlatiladi. allow_dangerous_deserialization=True parametri xavfsizlik choralarini biroz yumshatadi, chunki ba'zi eski FAISS bazalari yangi versiyalar bilan mos kelmasligi mumkin.

# 4. LLM VA PROMPT SOZLAMALARI (Llama 3.1)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4) # Badiiy asar uchun temp biroz ko'tarildi

system_prompt = (
    "Siz Tohir Malikning 'Ajab dunyo' kitobi bo'yicha mukammal javob beradigan adabiyotshunos yordamchisiz.\n"
    "Faqat va faqat quyida berilgan kontekst (kitob parchalari) ma'lumotlariga tayanib savolga javob bering.\n"
    "Javobingiz samimiy va chiroyli adabiy tilda bo'lsin. Agar javob kontekstda bo'lmasa, uni o'zingizdan to'qimang.\n\n"
    "Kontekst:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. RAG ZANJIRINI QURISH
# Kitob keng bo'lgani uchun savolga eng yaqin 3 ta chunkni qidiradigan qilamiz
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) # Qidiruv natijasida olingan chunklarni yagona matn ko'rinishida birlashtiradi. Har bir chunk orasiga ikki qator bo'sh joy qo'yiladi, bu esa javob berishda aniqroq va o'qilishi osonroq kontekst yaratadi.

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()} # Savol kelganda, retriever yordamida eng yaqin 3 ta chunkni qidirib olib, ularni format_docs funksiyasi orqali bitta matn holatiga keltiradi. Foydalanuvchi savoli esa o'zgarishsiz (passthrough) tarzda saqlanadi.
    | prompt # Tizim yo'riqnomasi va foydalanuvchi savolini birlashtirib, yakuniy xabar paketini tayyorlaydi.
    | llm # Llama 3.1 modeli bu xabar paketini olib, kontekstga asoslangan javob yaratadi. Bu model badiiy asarlar bo'yicha savollarga aniq va ma'noli javoblar berish uchun juda mos keladi.
    | StrOutputParser() # LLM dan olingan javobni oddiy matn formatida chiqarish uchun parser. Bu qism javobni tozalash va uni foydalanuvchiga aniq ko'rsatish uchun zarur.
)

# 6. MULOQOT INTERFEYSI
print("\n==================================================")
print("--- 'Ajab dunyo' kitobi bo'yicha FAISS RAG Agent! ---")
print("==================================================")

# print("\n=======================================================")
# print("LANGCHAIN INTERNET BOTI: SAVOLLARINIGIZNI YOZISHINGIZ MUMKIN!")
# print("CHIQISH UCHUN 'exit' DEB YOZING.")
# print("=======================================================")

while True:
    query = input("\nKitob bo'yicha savolingizni bering (Chiqish uchun 'exit'): ")
    if query.lower() == 'exit':
        break
    if query.strip() == "":
        continue
        
    print("Kitob sahifalari varaqlanmoqda...")
    try:
        response = rag_chain.invoke(query)
        print("\n[JAVOB]:", response)
    except Exception as e:
        print("\nXatolik yuz berdi:", e)
    finally:
            # Bu blok try to'g'ri ishlasa ham, xato bersa ham baribir ishga tushadi
            print("\n" + "_"*60 + "\n")










