import streamlit as st
# Sahifada tugmalar, slayderlar va natijalarni ko'rsatadigan veb-interfeysni yaratadi.
import joblib
# Avvaldan o'qitib saqlangan AI modelini (.pkl) xotiraga yuklaydi.
import pandas as pd
# Ma'lumotlarni model tushunadigan jadval (DataFrame) ko'rinishiga keltiradi.
import os
# Fayllar tizimi bilan ishlaydi, model qayerda turganini aniqlaydi.

# 1. Fayllarni yuklash
@st.cache_resource
# Bu maxsus buyruq bo'lib, modelni faqat bir marta yuklaydi. Sahifa har yangilanganda modelni qayta o'qib vaqt sarflamaslik uchun kerak.
def load_assets():
    base_path = os.path.dirname(__file__)  # Bu kod, joriy fayl (churn_app.py) joylashgan katalogni aniqlaydi. Bu, model va scaler fayllarining to'g'ri joylashganligini ta'minlash uchun kerak.
    model = joblib.load(os.path.join(base_path, 'churn_model.pkl')) # 'churn_model.pkl' faylini yuklaydi. Bu fayl, avval o'qitilgan va saqlangan AI modelini o'z ichiga oladi.
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl')) # 'scaler.pkl' faylini yuklaydi. Bu fayl, modelga kiritiladigan ma'lumotlarni o'lchamini moslashtirish uchun ishlatiladigan o'lchov vositasini o'z ichiga oladi.
    return model, scaler # Yuklangan model va scaler obyektlarini qaytaradi.

model, scaler = load_assets() # Yuklangan model va scaler obyektlarini o'zgaruvchilarga saqlaydi, shunda ular keyinchalik tahlil qilish jarayonida ishlatiladi.

# 2. Interfeys
st.title("🏦 Bank Churn Predictor") 
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Kredit balli", 300, 850, 650)
    age = st.slider("Yoshi", 18, 95, 35)
    tenure = st.slider("Bankdagi yillari", 0, 10, 5)
    balance = st.number_input("Balansi ($)", 0.0, 300000.0, 50000.0)
    num_products = st.selectbox("Mahsulotlar soni", [1, 2, 3, 4])

with col2:
    salary = st.number_input("Yillik maoshi ($)", 0.0, 250000.0, 100000.0)
    gender = st.selectbox("Jinsi", ["Female", "Male"])
    geography = st.selectbox("Davlat", ["France", "Germany", "Spain"])
    has_card = st.radio("Kredit kartasi bormi?", ["Ha", "Yo'q"])
    is_active = st.radio("Faol mijozmi?", ["Ha", "Yo'q"])
    marital = st.selectbox("Oilaviy holati", ["Married", "Single", "Divorced"])

# 3. Tahlil qilish
if st.button("Tahlil qilish"):
    # Barcha kerakli ustunlarni yaratamiz
    data = {
        'CreditScore': credit_score, 'Age': age, 'Tenure': tenure, 'Balance': balance,
        'NumOfProducts': num_products, 'HasCrCard': 1 if has_card == "Ha" else 0,
        'IsActiveMember': 1 if is_active == "Ha" else 0, 'EstimatedSalary': salary,
        'Complain': 0, 'Satisfaction Score': 3, 'Card Type': 1, 'Point Earned': 500,
        'DaySinceLastOrder': 5, 'CashbackAmount': 100,
        'Geography_France': 1 if geography == "France" else 0,
        'Geography_Germany': 1 if geography == "Germany" else 0,
        'Geography_Spain': 1 if geography == "Spain" else 0,
        'Gender_Female': 1 if gender == "Female" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'MaritalStatus_Divorced': 1 if marital == "Divorced" else 0,
        'MaritalStatus_Married': 1 if marital == "Married" else 0,
        'MaritalStatus_Single': 1 if marital == "Single" else 0
    }
    
    input_df = pd.DataFrame([data])

    # Model kutayotgan tartibni modelning o'zidan olamiz
    try:
        # XGBoost modelidan ustunlar tartibini olish
        model_features = model.get_booster().feature_names
        # Ma'lumotlarni o'sha tartibga majburiy o'tkazish
        input_df = input_df[model_features]
        
        # Bashorat
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)[0][1]

        if prediction[0] == 1:
            st.error(f"⚠️ Ketish ehtimoli: {prob:.1%}")
        else:
            st.success(f"✅ Mijoz qolishi kutilmoqda. (Ehtimol: {prob:.1%})")
            
    except Exception as e:
        # Agar get_booster ishlamasa (model sklearn wrapper bo'lsa), alifbo tartibini sinaymiz
        input_df = input_df.reindex(sorted(input_df.columns), axis=1)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.write("Natija (Alifbo tartibi bo'yicha chiqdi):", prediction[0])


# bu veb-ilova qanday vazifani bajaradi, qanday asosda ishlaydi,
# qanday tuzilgan va foydalari haqida ma'lumot:

# Vazifasi: Bank yoki e-savdo mijozlarining ketib qolish xavfini
# oldindan aytib beruvchi aqlli kalkulyator.

# Asosi: 5630 ta mijoz ma'lumotlari asosida o'qitilgan, 94.4% aniqlikka ega sun'iy intellekt (XGBoost).

# Tuzilishi: Visual Studio va Streamlit yordamida yaratilgan bo'lib,
# murakkab kodlarni tushunarli vizual shaklda (tugmalar, natijalar) ko'rsatadi.

# Foydasi: Tadbirkorga qaysi mijozga e'tibor berish kerakligini
# (ketish ehtimoli yuqorilarni) ko'rsatib, zararni kamaytirishga yordam beradi.



# Ishga tushirish uchun terminalda quyidagi buyruqlarni bajarish kerak:

# cd churn_wep_app
# C:\Users\Owner\AppData\Local\Programs\Python\Python313\python.exe -m streamlit run churn_app.py