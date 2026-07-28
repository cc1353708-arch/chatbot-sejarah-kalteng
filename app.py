import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import pickle
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

# ===================================================================
# 🎨 KONFIGURASI HALAMAN STREAMLIT
# ===================================================================
st.set_page_config(
    page_title="Chatbot Sejarah Kalimantan Tengah",
    page_icon="🏛️",
    layout="wide"
)

# ===================================================================
# 📚 KAMUS NORMALISASI (SAMA PERSIS DENGAN SCRIPT PELATIHAN)
# ===================================================================
KAMUS_SEJARAH = {
    "palangkaraya": "Palangka Raya", "barsel": "Barito Selatan", "bartim": "Barito Timur",
    "Blanda": "Belanda", "barut": "Barito Utara", "cilik riwut": "Tjilik Riwut", 
    "t. riwut": "Tjilik Riwut", "mahir mahar": "Mahir Mahar", "a. yunus": "Ahmad Yunus", 
    "letkol": "Letnan Kolonel", "kalteng": "Kalimantan Tengah", 
    "prov kalteng": "Provinsi Kalimantan Tengah", "ibukota kalteng": "Ibu Kota Kalimantan Tengah"
}

KAMUS_UMUM = {
    "dmana": "di mana", "dimana": "di mana", "kmna": "ke mana", "kpn": "kapan", 
    "knp": "kenapa", "gmn": "bagaimana", "gimana": "bagaimana", "yg": "yang", 
    "dg": "dengan", "dgn": "dengan", "sm": "sama", "ama": "sama", "krn": "karena", 
    "krna": "karena", "kpd": "kepada", "utk": "untuk", "dri": "dari", "pd": "pada", 
    "dlm": "dalam", "diantara": "di antara", "udh": "sudah", "udah": "sudah", 
    "blm": "belum", "blum": "belum", "bgt": "banget", "bgtt": "banget", "gtu": "begitu", 
    "kya": "seperti", "kyk": "seperti", "kek": "seperti", "aja": "saja", "cm": "cuma", 
    "cma": "cuma", "jd": "jadi", "trs": "terus", "trus": "terus", "lg": "lagi", 
    "lgi": "lagi", "jg": "juga", "jga": "juga", "gk": "tidak", "gak": "tidak", 
    "ga": "tidak", "ngga": "tidak", "tdk": "tidak", "bkn": "bukan", "tp": "tapi", 
    "klo": "kalau", "klu": "kalau", "gw": "saya", "gue": "saya", "ane": "saya", 
    "lo": "kamu", "lu": "kamu", "nih": "ini", "tuh": "itu", "kan": "bukan", 
    "ya": "iya", "y": "iya"
}

# ===================================================================
# 🛠️ FUNGSI PREPROCESSING
# ===================================================================
def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    for lama, baru in sorted(KAMUS_SEJARAH.items(), key=lambda x: len(x[0]), reverse=True):
        text = re.sub(r'\b' + re.escape(lama) + r'\b', baru, text, flags=re.IGNORECASE)
    for lama, baru in sorted(KAMUS_UMUM.items(), key=lambda x: len(x[0]), reverse=True):
        text = re.sub(r'\b' + re.escape(lama) + r'\b', baru, text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===================================================================
# 🚀 LOAD MODEL & PRE-COMPUTE VEKTOR KB (Agar Global Retrieval Instan)
# ===================================================================
@st.cache_resource
def load_models_and_precompute():
    kb = pickle.load(open('knowledge_base.pkl', 'rb'))
    
    tfidf_vec = joblib.load('model_tfidf.pkl')
    svm_tfidf = joblib.load('model_svm_tfidf.pkl')
    
    w2v_model = Word2Vec.load('model_word2vec.pkl')
    svm_w2v = joblib.load('model_svm_w2v.pkl')
    
    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    svm_sbert = joblib.load('model_svm_sbert.pkl')
    
    # Pre-compute vektor untuk seluruh Knowledge Base (Hemat waktu saat chat)
    kb_questions = kb['question'].tolist()
    
    kb_tfidf = tfidf_vec.transform(kb_questions)
    
    kb_w2v = np.array([
        np.mean([w2v_model.wv[w] for w in q.split() if w in w2v_model.wv], axis=0)
        if any(w in w2v_model.wv for w in q.split()) else np.zeros(100) for q in kb_questions
    ]).reshape(len(kb_questions), -1)
    
    kb_sbert = sbert_model.encode(kb_questions)
    
    return (kb, tfidf_vec, svm_tfidf, w2v_model, svm_w2v, sbert_model, svm_sbert, 
            kb_tfidf, kb_w2v, kb_sbert)

with st.spinner("⏳ Memuat model AI & menghitung vektor Knowledge Base... Mohon tunggu sebentar..."):
    (kb, tfidf_vec, svm_tfidf, w2v_model, svm_w2v, sbert_model, svm_sbert, 
     kb_tfidf, kb_w2v, kb_sbert) = load_models_and_precompute()

# ===================================================================
# 🧠 FUNGSI CHATBOT (Mendukung Two-Stage & Global Retrieval)
# ===================================================================
THRESHOLD = 0.75

def get_response(user_input, method, mode):
    clean_text = preprocess_text(user_input)
    if not clean_text:
        return "⚠️ Maaf, input tidak valid.", "-", "-", 0.0, "-"
    
    # 1. Ekstraksi Fitur User
    if method == 'TF-IDF':
        user_vector = tfidf_vec.transform([clean_text])
        kb_vectors = kb_tfidf
    elif method == 'Word2Vec':
        words = clean_text.split()
        vec = np.mean([w2v_model.wv[w] for w in words if w in w2v_model.wv], axis=0) if any(w in w2v_model.wv for w in words) else np.zeros(100)
        user_vector = vec.reshape(1, -1)
        kb_vectors = kb_w2v
    elif method == 'SBERT':
        user_vector = sbert_model.encode([clean_text])
        kb_vectors = kb_sbert

    # 2. Tentukan Ruang Pencarian (Filter SVM atau Global)
    if mode == "Two-Stage (Dengan Filter SVM)":
        # Prediksi kategori dengan SVM
        if method == 'TF-IDF': predicted_category = svm_tfidf.predict(user_vector)[0]
        elif method == 'Word2Vec': predicted_category = svm_w2v.predict(user_vector)[0]
        elif method == 'SBERT': predicted_category = svm_sbert.predict(user_vector)[0]
        
        # Filter KB hanya untuk kategori yang diprediksi
        kb_mask = (kb['kategori'] == predicted_category).to_numpy()
        filtered_kb_vectors = kb_vectors[kb_mask]
        filtered_kb_df = kb[kb_mask]
        
        if filtered_kb_df.empty:
            return f"⚠️ Tidak ada data untuk kategori '{predicted_category}'.", predicted_category, "-", 0.0, "-"
            
        similarities = cosine_similarity(user_vector, filtered_kb_vectors)
        best_idx_local = similarities.argmax()
        best_score = similarities[0][best_idx_local]
        
        matched_question = filtered_kb_df.iloc[best_idx_local]['question']
        kb_category = filtered_kb_df.iloc[best_idx_local]['kategori']
        best_answer = filtered_kb_df.iloc[best_idx_local]['answer']
        
    else: # Global (Tanpa SVM)
        predicted_category = "N/A (Mode Global)"
        
        similarities = cosine_similarity(user_vector, kb_vectors)
        best_idx = similarities.argmax()
        best_score = similarities[0][best_idx]
        
        matched_question = kb.iloc[best_idx]['question']
        kb_category = kb.iloc[best_idx]['kategori']
        best_answer = kb.iloc[best_idx]['answer']

    # 3. Threshold Check
    if best_score < THRESHOLD:
        return (f"⚠️ Maaf, saya belum memiliki informasi yang cukup spesifik untuk pertanyaan tersebut. "
                f"(Skor kemiripan: {best_score:.2f})", 
                predicted_category, matched_question, best_score, kb_category)
    
    return best_answer, predicted_category, matched_question, best_score, kb_category

# ===================================================================
# 🎨 TAMPILAN UTAMA (UI)
# ===================================================================
st.title("🏛️ Chatbot Sejarah Kalimantan Tengah")
st.markdown("""
Selamat datang! Chatbot ini dapat menjawab pertanyaan seputar **sejarah lokal Kalimantan Tengah**, 
mulai dari tokoh, peristiwa, budaya, hingga kondisi politik masa lalu.
""")
st.divider()

# Sidebar - Informasi & Pengaturan
with st.sidebar:
    st.header("⚙️ Pengaturan Eksperimen")
    
    st.markdown("**1. Pilih Metode Representasi Teks:**")
    selected_methods = st.multiselect(
        "Metode yang ingin ditampilkan:",
        options=["TF-IDF", "Word2Vec", "SBERT"],
        default=["TF-IDF", "Word2Vec", "SBERT"],
        help="Pilih satu atau lebih metode untuk dibandingkan secara head-to-head."
    )
    
    st.divider()
    
    st.markdown("**2. Pilih Mode Retrieval:**")
    retrieval_mode = st.radio(
        "Arsitektur Pencarian:",
        ["Two-Stage (Dengan Filter SVM)", "Global (Tanpa SVM)"],
        help="Two-Stage: SVM menebak kategori, lalu mencari di kategori tersebut. Global: Mencari di seluruh Knowledge Base tanpa filter kategori."
    )
    
    st.divider()
    st.markdown("### 📊 Informasi Model")
    st.markdown(f"""
    - **Knowledge Base:** {len(kb)} pertanyaan
    - **Kategori Intent:** {kb['kategori'].nunique()} kategori
    - **Threshold Similarity:** {THRESHOLD}
    - **Mode Aktif:** `{retrieval_mode}`
    """)
    
    st.divider()
    st.markdown("### 💡 Contoh Pertanyaan")
    contoh_pertanyaan = [
        "Siapa itu Tjilik Riwut?",
        "Kapan Pertempuran Danau Mare terjadi?",
        "Apa itu GRRI?",
        "Bagaimana sistem kekerabatan suku Dayak?",
        "Kapan Jepang masuk ke Kalimantan Tengah?"
    ]
    for q in contoh_pertanyaan:
        st.markdown(f"- *{q}*")

# ===================================================================
# 💬 AREA CHAT
# ===================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ketik pertanyaan sejarah Kalimantan Tengah di sini...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    responses = {}
    for method in selected_methods:
        # Panggil fungsi dengan parameter 'mode'
        answer, svm_category, matched_q, score, kb_category = get_response(user_input, method, retrieval_mode)
        responses[method] = {
            "answer": answer,
            "svm_category": svm_category,
            "matched_q": matched_q,
            "score": score,
            "kb_category": kb_category
        }
    
    st.session_state.chat_history.append({"role": "bot", "content": responses})

# Tampilkan riwayat chat
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(f"**{msg['content']}**")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            responses = msg["content"]
            
            if len(responses) > 1:
                cols = st.columns(len(responses))
                for idx, (method, data) in enumerate(responses.items()):
                    with cols[idx]:
                        color = "🟢" if method == "TF-IDF" else ("🟡" if method == "Word2Vec" else "🔵")
                        st.markdown(f"#### {color} {method}")
                        st.markdown(data["answer"])
                        
                        with st.expander("🔍 Detail Teknis"):
                            if "Two-Stage" in retrieval_mode:
                                st.markdown(f"- **Kategori Tebakan SVM:** `{data['svm_category']}`")
                            st.markdown(f"- **Kategori Soal KB Terpilih:** `{data['kb_category']}`")
                            st.markdown(f"- **Skor Cosine:** `{data['score']:.4f}`")
                            st.markdown(f"- **Soal KB Terpilih:** *{data['matched_q']}*")
            else:
                method = list(responses.keys())[0]
                data = responses[method]
                st.markdown(data["answer"])
                
                with st.expander("🔍 Detail Teknis"):
                    if "Two-Stage" in retrieval_mode:
                        st.markdown(f"- **Kategori Tebakan SVM:** `{data['svm_category']}`")
                    st.markdown(f"- **Kategori Soal KB Terpilih:** `{data['kb_category']}`")
                    st.markdown(f"- **Skor Cosine:** `{data['score']:.4f}`")
                    st.markdown(f"- **Soal KB Terpilih:** *{data['matched_q']}*")

# Tombol hapus riwayat
if st.session_state.chat_history:
    st.divider()
    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.chat_history = []
        st.rerun()