import streamlit as st
import pandas as pd
import joblib
import os

# 1️⃣ Model ve encoder yükle
model = joblib.load("xgboost_model.pkl")
encoder = joblib.load("encoder.pkl")

st.title("🎨 XGBoost Tahmin Uygulaması")
st.write("Aşağıdaki değerleri girerek tahminleme yapabilirsiniz.")

# 2️⃣ Kullanıcıdan verileri al
with st.form("prediction_form"):
    st.subheader("Girdi Değerleri (X)")
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

    st.subheader("Gerçek Y Değerleri (isteğe bağlı)")
    bicak_aniloks_mesafe_real = st.number_input("Bıçak-Aniloks Mesafe (gerçek)")
    aniloks_klise_mesafe_real = st.number_input("Aniloks-Klişe Mesafe (gerçek)")
    klise_tambur_mesafe_real = st.number_input("Klişe-Tambur Mesafe (gerçek)")

    submitted = st.form_submit_button("🔍 Tahmin Et")

# 3️⃣ Tahmin ve kayıt
if submitted:
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

    df_new = pd.DataFrame([data])

    # Encode kategorik
    encoded_cat = encoder.transform(df_new[["aniloks_no", "klise_no"]])
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(["aniloks_no", "klise_no"]))
    numeric_new_df = df_new.drop(columns=["aniloks_no", "klise_no"])
    df_new_encoded = pd.concat([encoded_cat_df, numeric_new_df], axis=1)

    model_features = model.estimators_[0].get_booster().feature_names
    df_new_encoded = df_new_encoded[model_features]

    # Tahmin
    prediction = model.predict(df_new_encoded)

    st.success("✅ Tahmin tamamlandı!")
    st.subheader("📊 Tahmin Sonuçları")
    st.write(f"**Bıçak-Aniloks Mesafe:** {prediction[0][0]:.2f}")
    st.write(f"**Aniloks-Klişe Mesafe:** {prediction[0][1]:.2f}")
    st.write(f"**Klişe-Tambur Mesafe:** {prediction[0][2]:.2f}")

    # Sonuçları Excel'e kaydet
    result_row = df_new.copy()
    result_row["bicak_aniloks_mesafe_pred"] = prediction[0][0]
    result_row["aniloks_klise_mesafe_pred"] = prediction[0][1]
    result_row["klise_tambur_mesafe_pred"] = prediction[0][2]

    # Kullanıcının girdiği gerçek Y değerleri
    result_row["bicak_aniloks_mesafe_real"] = bicak_aniloks_mesafe_real
    result_row["aniloks_klise_mesafe_real"] = aniloks_klise_mesafe_real
    result_row["klise_tambur_mesafe_real"] = klise_tambur_mesafe_real

    if os.path.exists("results.xlsx"):
        old_df = pd.read_excel("results.xlsx")
        updated_df = pd.concat([old_df, result_row], ignore_index=True)
    else:
        updated_df = result_row

    updated_df.to_excel("results.xlsx", index=False)

# 6️⃣ Yönetici için indirilebilir dosya
if os.path.exists("results.xlsx"):
    with open("results.xlsx", "rb") as f:
        st.download_button(
            label="📥 Kaydedilen Sonuçları İndir (Excel)",
            data=f,
            file_name="tahmin_sonuclari.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )



