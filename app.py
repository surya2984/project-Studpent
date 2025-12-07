import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import base64
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# --- FUNGSI LOAD CSS EKSTERNAL ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"File {file_name} tidak ditemukan.")

local_css("style.css")

# --- FUNGSI BACKGROUND ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        encoded_bg = get_base64_image(image_file)
        page_bg = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_bg}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp { background: linear-gradient(to right, #e0c3fc, #8ec5fc); }
        </style>
        """, unsafe_allow_html=True)

add_bg_from_local("ungu.jpg")

# --- LOAD MODEL ---
MODEL_PATH = "best_mobilenetv2.keras"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    else:
        return None

model = load_model()

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (200, 200)

# --- FUNGSI PREDIKSI ---
def predict(image):
    img = image.resize(IMG_SIZE)
    img = img.convert("RGB")
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    if model:
        pred = model.predict(img_array)
        class_id = np.argmax(pred)
        confidence = np.max(pred)
        return CLASS_NAMES[class_id], pred[0]
    else:
        return "Model Error", [0, 0, 0, 0]

# --- SIDEBAR MENU ---
with st.sidebar:
    st.title("🧭 Petunjuk Menu")
    menu = st.radio("Pilih Menu", [
        "Klasifikasi Hasil MRI",
        "Tentang Tumor Otak",
        "Tips Kesehatan Otak",
        "Tentang Aplikasi"
    ])
    st.markdown("---")
    st.info("Aplikasi ini menggunakan AI untuk mendeteksi pola pada citra MRI.")

# --- HALAMAN UTAMA ---

if menu == "Klasifikasi Hasil MRI":
    st.title("🧠 Brain Tumor Classification")
    st.markdown("### Deteksi Dini Menggunakan Deep Learning")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("Silahkan upload gambar MRI kepala untuk mendeteksi jenis tumor.")
        uploaded_file = st.file_uploader("Upload gambar MRI (jpg/png)", type=["jpg", "jpeg", "png"])

    with col2:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Diunggah", width=300)

            if st.button("🔍 Analisis Gambar"):
                if model is None:
                    st.error("Model tidak ditemukan.")
                else:
                    with st.spinner('Sedang menganalisis gambar...'):
                        label, probabilities = predict(image)

                    st.markdown("---")
                    st.subheader(f"🧾 Hasil Prediksi: **{label.upper()}**")

                    st.write("Probabilitas Deteksi:")
                    for cls, prob in zip(CLASS_NAMES, probabilities):
                        col_name, col_bar, col_val = st.columns([1, 3, 1])
                        with col_name:
                            st.write(f"**{cls.capitalize()}**")
                        with col_bar:
                            st.progress(float(prob))
                        with col_val:
                            st.write(f"{prob*100:.2f}%")

                    st.markdown("---")
                    st.markdown("### 🩺 Penjelasan Medis:")

                    if label == "notumor":
                        st.success("✔ **Tidak terdeteksi adanya tumor.**")
                        st.info("Tetap konsultasikan ke dokter untuk hasil akurat.")
                    else:
                        st.warning(f"⚠ Terindikasi adanya **{label.capitalize()} Tumor**.")
                        st.info("Hasil ini bukan diagnosis final. Segera konsultasikan ke dokter.")

elif menu == "Tentang Tumor Otak":
    st.title("📚 Edukasi Tumor Otak")
    
    tab1, tab2, tab3 = st.tabs(["Glioma", "Meningioma", "Pituitary"])
    
    with tab1:
        st.header("Glioma")
        st.write("Tumor yang tumbuh pada sel glial otak atau sel pendukung otak.")
        st.warning("Jenis ini seringkali tumbuh di jaringan otak itu sendiri.")
        
    with tab2:
        st.header("Meningioma")
        st.write("Tumbuh pada meninges atau selaput otak yang mengelilingi bagian luar otak.")
        st.info("Seringkali jinak tetapi dapat menekan otak jika membesar.")
        
    with tab3:
        st.header("Pituitary")
        st.write("Tumbuh pada kelenjar pituitari atau kelenjar hormon di dasar otak.")
        st.success("Dapat mempengaruhi keseimbangan hormon tubuh.")

elif menu == "Tips Kesehatan Otak":
    st.header("🧘 Tips Menjaga Kesehatan Otak")
    st.write("Berikut tips sederhana untuk menjaga fungsi otak:")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🍎 1. Nutrisi")
        st.write("Konsumsi Blueberry, kacang-kacangan, ikan salmon, dan alpukat.")
        st.subheader("🏃 2. Olahraga")
        st.write("Olahraga ringan 20–30 menit per hari meningkatkan aliran darah.")
        st.subheader("🧠 3. Kognitif")
        st.write("Membaca buku, bermain puzzle, menghafal.")
    with col_b:
        st.subheader("😴 4. Istirahat")
        st.write("Tidur 7–9 jam penting untuk memori.")
        st.subheader("💧 5. Hidrasi")
        st.write("Kurang minum membuat otak sulit berkonsentrasi.")
        st.subheader("🧘 6. Anti Stres")
        st.write("Meditasi ringan atau aktivitas menenangkan.")
    
    st.success("💡 *Kebiasaan sehat sehari-hari dapat membantu menjaga kesehatan otak.*")

elif menu == "Tentang Aplikasi":
    st.header("ℹ Tentang Aplikasi")
    
    
    st.markdown("""
<div class="about-box">
    <p>Aplikasi ini dibuat untuk melakukan deteksi awal tumor otak menggunakan model <b>Deep Learning (MobileNetV2)</b>.</p>
    <h4>Fitur utama:</h4>
    <ul>
        <li>Upload MRI & prediksi otomatis</li>
        <li>Visualisasi probabilitas akurasi</li>
        <li>Edukasi jenis tumor & kesehatan</li>
    </ul>
    <hr style="border-top: 1px solid #ddd;">
    <small><i>Disclaimer: Aplikasi ini bukan alat diagnosis resmi dan akurat 100%. Ini hanya alat bantu edukasi dan deteksi awal. Selalu konsultasikan dengan dokter ahli.</i></small>
</div>
""", unsafe_allow_html=True)