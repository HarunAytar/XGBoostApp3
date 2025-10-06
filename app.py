import streamlit as st
import pandas as pd
import joblib

# 🎯 1️⃣ Model ve encoder yükle
model = joblib.load("xgboost_model.pkl")
encoder = joblib.load("encoder.pkl")

st.title("🎨 XGBoost Tahmin Uygulaması")
st.write("Aşağıdaki değerleri girerek tahminleme yapabilirsiniz.")

# 🎯 2️⃣ Kullanıcıdan verileri al
with st.form("prediction_form"):
    st.subheader("Girdi Değerleri")

    aniloks_no = st.number_input("Aniloks numarası", min_value=0, step=1)
    klise_no = st.number_input("Klişe numarası", min_value=0, step=1)
    aniloks_aktarma = st.number_input("Aniloks aktarma değeri")
    klise_tıram_oranı = st.number_input("Klişe tıram oranı")
    siliv_capı = st.number_input("Siliv çapı")
    tesa_esneme = st.number_input("Tesa esneme")
    hiz = st.number_input("Hız")
    basılacak_film_uzunluk = st.number_input("Basılacak film uzunluğu")
    hazırlanan_boya_visko = st.number_input("Hazırlanan boya viskozitesi")
    referans_renk_L = st.number_input("Referans renk L")
    referans_renk_a = st.number_input("Referans renk a")
    referans_renk_b = st.number_input("Referans renk b")
    film_renk_L = st.number_input("Film renk L")
    film_renk_a = st.number_input("Film renk a")
    film_renk_b = st.number_input("Film renk b")
    film_seffaflık = st.number_input("Film şeffaflık")
    film_kalınlık = st.number_input("Film kalınlık")
    hazırlanan_boya_L = st.number_input("Hazırlanan boya L")
    hazırlanan_boya_a = st.number_input("Hazırlanan boya a")
    hazırlanan_boya_b = st.number_input("Hazırlanan boya b")

    submitted = st.form_submit_button("🔍 Tahmin Et")

# 🎯 3️⃣ Tahmin işlemi
if submitted:
    # Kullanıcıdan alınan verileri bir sözlükte topla
    data = {
        "aniloks_no": aniloks_no,
        "klise_no": klise_no,
        "aniloks_aktarma": aniloks_aktarma,
        "klise_tıram_oranı": klise_tıram_oranı,
        "siliv_capı": siliv_capı,
        "tesa_esneme": tesa_esneme,
        "hiz": hiz,
        "basılacak_film_uzunluk": basılacak_film_uzunluk,
        "hazırlanan_boya_visko": hazırlanan_boya_visko,
        "referans_renk_L": referans_renk_L,
        "referans_renk_a": referans_renk_a,
        "referans_renk_b": referans_renk_b,
        "film_renk_L": film_renk_L,
        "film_renk_a": film_renk_a,
        "film_renk_b": film_renk_b,
        "film_seffaflık": film_seffaflık,
        "film_kalınlık": film_kalınlık,
        "hazırlanan_boya_L": hazırlanan_boya_L,
        "hazırlanan_boya_a": hazırlanan_boya_a,
        "hazırlanan_boya_b": hazırlanan_boya_b
    }

    # DataFrame oluştur
    df_new = pd.DataFrame([data])

    # Kategorik değişkenleri encode et
    encoded_cat = encoder.transform(df_new[["aniloks_no", "klise_no"]])
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(["aniloks_no", "klise_no"]))

    # Sayısal sütunları koru
    numeric_new_df = df_new.drop(columns=["aniloks_no", "klise_no"])

    # Tüm sütunları birleştir
    df_new_encoded = pd.concat([encoded_cat_df, numeric_new_df], axis=1)

    # Modelin beklediği sıraya göre sırala
    model_features = model.get_booster().feature_names
    df_new_encoded = df_new_encoded[model_features]

    # Tahmin yap
    prediction = model.predict(df_new_encoded)

    # Sonuçları göster
    st.success("✅ Tahmin tamamlandı!")
    st.subheader("📊 Tahmin Sonuçları")
    st.write(f"**Bıçak-Aniloks Mesafe:** {prediction[0][0]:.2f}")
    st.write(f"**Aniloks-Klişe Mesafe:** {prediction[0][1]:.2f}")
    st.write(f"**Klişe-Tambur Mesafe:** {prediction[0][2]:.2f}")
