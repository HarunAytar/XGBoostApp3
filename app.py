import streamlit as st
import pandas as pd
import joblib

# Başlık
st.title("🎯 XGBoost Tahmin Uygulaması")

# 1️⃣ Model ve encoder yükle
model = joblib.load("xgboost_model.pkl")
encoder = joblib.load("encoder.pkl")

# 2️⃣ Kullanıcıdan yeni X değerlerini al
st.header("Yeni bir gözlem için X değerlerini gir:")

data = {}
data["aniloks_no"] = st.number_input("Aniloks numarasını girin:", min_value=0, step=1)
data["klise_no"] = st.number_input("Klişe numarasını girin:", min_value=0, step=1)
data["aniloks_aktarma"] = st.number_input("Aniloks aktarma değeri:", step=0.01)
data["klise_tıram_oranı"] = st.number_input("Klişe tıram oranı:", step=0.01)
data["tesa_esneme"] = st.number_input("Tesa esneme değeri:", step=0.01)
data["hiz"] = st.number_input("Hız değeri:", step=0.01)
data["basılacak_film_uzunluk"] = st.number_input("Basılacak film uzunluğu:", step=0.01)
data["hazırlanan_boya_visko"] = st.number_input("Hazırlanan boya viskozitesi:", step=0.01)
data["referans_renk_L"] = st.number_input("Referans renk L değeri:", step=0.01)
data["referans_renk_a"] = st.number_input("Referans renk a değeri:", step=0.01)
data["referans_renk_b"] = st.number_input("Referans renk b değeri:", step=0.01)
data["film_renk_L"] = st.number_input("Film renk L değeri:", step=0.01)
data["film_renk_a"] = st.number_input("Film renk a değeri:", step=0.01)
data["film_renk_b"] = st.number_input("Film renk b değeri:", step=0.01)
data["film_seffaflık"] = st.number_input("Film şeffaflık değeri:", step=0.01)
data["film_kalınlık"] = st.number_input("Film kalınlık değeri:", step=0.01)
data["hazırlanan_boya_L"] = st.number_input("Hazırlanan boya L değeri:", step=0.01)
data["hazırlanan_boya_a"] = st.number_input("Hazırlanan boya a değeri:", step=0.01)
data["hazırlanan_boya_b"] = st.number_input("Hazırlanan boya b değeri:", step=0.01)

# 3️⃣ Tahmin butonu
if st.button("Tahmin Yap"):
    # DataFrame oluştur
    df_new = pd.DataFrame([data])

    # Kategorik değişkenleri encode et
    encoded_cat = encoder.transform(df_new[["aniloks_no", "klise_no"]])
    encoded_cat_df = pd.DataFrame(
        encoded_cat, 
        columns=encoder.get_feature_names_out(["aniloks_no", "klise_no"])
    )

    # Sayısal sütunları koru
    numeric_new_df = df_new.drop(columns=["aniloks_no", "klise_no"])

    # Tüm sütunları birleştir
    df_new_encoded = pd.concat([encoded_cat_df, numeric_new_df], axis=1)

    # Sütunları modelin beklediği sıraya göre sırala
    model_features = model.estimators_[0].get_booster().feature_names
    df_new_encoded = df_new_encoded[model_features]

    # Tahmin yap
    prediction = model.predict(df_new_encoded)

    # 4️⃣ Sonuçları göster
    st.subheader("🎯 Tahmin Sonuçları")
    st.write(f"**bıçak_aniloks_mesafe:** {prediction[0][0]:.2f}")
    st.write(f"**aniloks_klişe_mesafe:** {prediction[0][1]:.2f}")
    st.write(f"**klişe_tambur_mesafe:** {prediction[0][2]:.2f}")
