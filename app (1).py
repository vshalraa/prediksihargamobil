
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Harga Mobil Bekas",
    page_icon="🚗",
    layout="centered"
)

# --- FUNGSI LOAD DATA & MODEL ---
@st.cache_resource
def load_model():
    """Memuat model Machine Learning yang sudah dilatih"""
    try:
        with open('Adaboost.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

@st.cache_data
def load_car_names():
    """Memuat daftar nama mobil dan mapping ID-nya"""
    try:
        # Membaca file Car Model ID.csv
        df = pd.read_csv('Car Model ID.csv')
        
        # Asumsi: File ini berisi kolom 'Jenis mobil' atau sejenisnya.
        # Kita akan membuat dictionary: {Nama Mobil: ID}
        # ID diasumsikan berdasarkan index baris + 1 (atau sesuai format training Anda)
        # Jika training dimulai dari 0, hapus '+ 1'. Mari kita default ke index (0-based) atau (1-based)
        # Berdasarkan dataset umum, ID seringkali dimulai dari integer tertentu.
        # Disini kita mapping index dataframe ke Nama.
        
        car_dict = {row[0]: idx for idx, row in df.iterrows()}
        return car_dict
    except Exception as e:
        st.error(f"Gagal memuat daftar mobil: {e}")
        return {}

# --- LOAD RESOURCES ---
model = load_model()
car_map = load_car_names()

# --- UI HEADER ---
st.title("🚗 Aplikasi Prediksi Harga Mobil")
st.markdown("""
Aplikasi ini menggunakan algoritma **AdaBoost** untuk memprediksi harga pasaran mobil bekas 
berdasarkan spesifikasi dan fitur yang dimiliki.
""")
st.divider()

# --- FORM INPUT USER ---
with st.form("prediction_form"):
    st.subheader("Spesifikasi Mobil")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Input Nama Mobil
        # Mengambil list nama mobil dari mapping yang sudah di-load
        car_name_options = list(car_map.keys()) if car_map else ["Data Kosong"]
        selected_car_name = st.selectbox("Pilih Model Mobil", car_name_options)
        
        # Input Tahun
        year = st.number_input("Tahun Pembuatan", min_value=2000, max_value=2025, value=2018, step=1)
        
    with col2:
        # Input Transmisi
        # Perlu dipastikan encoding saat training. 
        # Umumnya: 1 = Automatic, 0 = Manual (atau sebaliknya).
        # Di sini saya asumsikan 1 = Automatic, 0 = Manual.
        transmission_option = st.radio("Transmisi", ["Manual", "Automatic"])
        transmission = 0 if transmission_option == "Automatic" else 1

    st.subheader("Fitur Tambahan")
    st.caption("Centang fitur yang tersedia pada mobil:")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        sunroof = st.checkbox("Sun Roof")
        auto_retract = st.checkbox("Auto Retract Mirror")
        
    with c2:
        electric_parking = st.checkbox("Electric Parking Brake")
        vehicle_stability = st.checkbox("Vehicle Stability Control")
        
    with c3:
        auto_cruise = st.checkbox("Auto Cruise Control")

    # Tombol Submit
    submitted = st.form_submit_button("🔍 Prediksi Harga", use_container_width=True)

# --- LOGIKA PREDIKSI ---
if submitted:
    if model is None:
        st.error("Model belum dimuat, tidak bisa melakukan prediksi.")
    elif not car_map:
        st.error("Data mapping mobil kosong.")
    else:
        # 1. Konversi Input ke Format Model
        # Mendapatkan ID mobil dari nama yang dipilih
        car_id = car_map[selected_car_name]
        
        # Konversi boolean ke int (0/1)
        feat_sunroof = 1 if sunroof else 0
        feat_retract = 1 if auto_retract else 0
        feat_electric = 1 if electric_parking else 0
        feat_vsc = 1 if vehicle_stability else 0
        feat_cruise = 1 if auto_cruise else 0
        
        # Urutan fitur HARUS SAMA PERSIS dengan saat training model (X_train)
        # Berdasarkan file pkl: ['year', 'car name', 'transmission', 'sun roof', 'auto retract mirror', 'electric parking brake', 'vehicle stability control', 'auto cruise control']
        
        input_data = np.array([[
            year,
            car_id,
            transmission,
            feat_sunroof,
            feat_retract,
            feat_electric,
            feat_vsc,
            feat_cruise
        ]])
        
        # 2. Lakukan Prediksi
        try:
            predicted_price = model.predict(input_data)[0]
            
            # 3. Tampilkan Hasil
            st.success("Prediksi Selesai!")
            st.metric(
                label="Estimasi Harga Jual", 
                value=f"Rp {int(predicted_price):,.0f}".replace(",", ".")
            )
            st.info("Catatan: Harga ini adalah estimasi berdasarkan data historis. Kondisi fisik aktual mobil dapat mempengaruhi harga nyata.")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memprediksi: {e}")

# --- SIDEBAR (Optional) ---
st.sidebar.header("Tentang Aplikasi")
st.sidebar.info("Dibuat menggunakan Python & Streamlit dengan model Machine Learning AdaBoost Regressor.")
