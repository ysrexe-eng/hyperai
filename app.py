import os

# 1. Streamlit watcher & PyTorch thread optimizasyonları (Sistem kasmasını önler)
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import json
import re
import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from google.genai.errors import APIError

torch.set_num_threads(2)

# ---------------------------------------------------------------
# 2. Sayfa Yapılandırması ve Stiller
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Hyprland Config Asistanı",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header { font-size: 2rem; font-weight: 700; color: #58a6ff; margin-bottom: 0.2rem; }
    .sub-header { font-size: 0.95rem; color: #8b949e; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

GEN_MODEL = "gemma-4-31b-it"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
INPUT_PATH = "hyprland_dataset.json"
CACHE_PATH = "rag_cache.npz"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 120
SKIP_KEYWORDS = ["readme", "version-selector", "_index", "license"]

# ---------------------------------------------------------------
# 3. API Key Yönetimi (Arayüzde GÖSTERİLMEZ)
# ---------------------------------------------------------------
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")

# Sol Yan Panel
with st.sidebar:
    st.title("⚡ Hyprland RAG")
    st.markdown("---")
    
    top_k_slider = st.slider("Getirilecek Doküman (Top K)", min_value=3, max_value=20, value=10)
    temp_slider = st.slider("Sıcaklık (Temperature)", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    use_expansion = st.checkbox("Gelişmiş Arama (Query Expansion)", value=False, help="LLM ile ekstra ayar isimleri tahmin eder. Aramayı bir tık yavaşlatabilir.")
    
    st.markdown("---")
    st.markdown("### 📊 Durum")
    status_box = st.empty()

# ---------------------------------------------------------------
# 4. Önbelleğe Alınmış Yüklemeler
# ---------------------------------------------------------------
@st.cache_resource
def get_gemini_client(key):
    if not key:
        return None
    return genai.Client(api_key=key)

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_data
def load_data_and_cache():
    def parse_tables(text):
        rows = []
        for line in text.split("\n"):
            line = line.strip()
            if not line.startswith("|") or line.count("|") < 3:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r"[-: ]*", c) for c in cells) or (cells and cells[0].lower() in ("name", "key", "setting")):
                continue
            if len(cells) >= 2 and cells[0]:
                rows.append(f"Ayar: {cells[0]}\n" + "\n".join(f"{h}: {v}" for h, v in zip(["Açıklama", "Tip", "Varsayılan"], cells[1:])))
        return rows

    def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start+size])
            start += size - overlap
        return chunks

    if not os.path.exists(INPUT_PATH):
        st.error(f"❌ '{INPUT_PATH}' bulunamadı!")
        st.stop()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    documents = []
    for item in raw_data:
        topic = item["instruction"].split(":", 1)[-1].strip()
        if any(kw in topic.lower() for kw in SKIP_KEYWORDS):
            continue

        for row in parse_tables(item["output"]):
            documents.append({"topic": topic, "text": row, "is_table_row": True})

        for chunk in chunk_text(item["output"]):
            if len(chunk.strip()) >= 50:
                documents.append({"topic": topic, "text": chunk, "is_table_row": False})

    if not os.path.exists(CACHE_PATH):
        st.error(f"❌ Önbellek dosyası '{CACHE_PATH}' bulunamadı! Lütfen GitHub repana ekle.")
        st.stop()

    cache = np.load(CACHE_PATH, allow_pickle=True)
    embeddings = cache["embeddings"]

    return documents, embeddings

# Başlatma
if not API_KEY:
    status_box.error("❌ GEMINI_API_KEY bulunamadı!")
    st.error("🔑 API Key eksik. Lütfen Streamlit Cloud Secrets alanına `GEMINI_API_KEY` tanımını ekleyin.")
    st.stop()

try:
    embedder = load_embedder()
    documents, embeddings = load_data_and_cache()
    client = get_gemini_client(API_KEY)
    status_box.success(f"✅ Hazır! ({len(embeddings)} embedding aktif)")
except Exception as e:
    status_box.error(f"Sistem Hatası: {e}")
    st.stop()

# ---------------------------------------------------------------
# 5. Arama Fonksiyonları
# ---------------------------------------------------------------
def expand_query(question):
    prompt = f"Hyprland config sorusu: '{question}'. İlgili config parametrelerini İngilizce, virgülle ayırarak yaz (örn: border_size, gaps_in)."
    try:
        res = client.models.generate_content(
            model=GEN_MODEL, 
            contents=prompt, 
            config=types.GenerateContentConfig(temperature=0.2)
        )
        return [t.strip() for t in res.text.split(",") if t.strip()]
    except Exception:
        return []

def retrieve(question, top_k, enable_expansion):
    expanded = expand_query(question) if enable_expansion else []
    q_text = "query: " + question + (" " + " ".join(expanded) if expanded else "")
    
    q_emb = embedder.encode([q_text], convert_to_numpy=True)[0]
    q_emb = q_emb / np.linalg.norm(q_emb)
    scores = embeddings @ q_emb

    keywords = set(re.findall(r'\b[a-zA-Z_]+(?:\.[a-zA-Z_]+)+\b', question) + re.findall(r'\b[a-zA-Z]+_[a-zA-Z_]+\b', question) + expanded)
    
    exact_match_idx = []
    if keywords:
        for i, doc in enumerate(documents):
            if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', doc["text"].lower()) for kw in keywords if len(kw) > 2):
                exact_match_idx.append(i)

    exact_match_idx.sort(key=lambda i: scores[i], reverse=True)
    candidate_idx = exact_match_idx + [i for i in np.argsort(scores)[::-1] if i not in set(exact_match_idx)]

    final_idx = []
    for i in candidate_idx:
        if len(final_idx) >= top_k:
            break
        txt = documents[i]["text"]
        if not any(txt[:100] in documents[j]["text"] for j in final_idx):
            final_idx.append(i)

    return [(documents[i], scores[i]) for i in final_idx], expanded

# ---------------------------------------------------------------
# 6. Chat Arayüzü & Streaming
# ---------------------------------------------------------------
st.markdown('<div class="main-header">Hyprland Config Asistanı</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Canlı yanıt akışlı sohbet asistanı.</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş Sohbet Ekranı
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("🔍 Yanıt İçin Kullanılan Kaynaklar"):
                for doc, score in msg["sources"]:
                    tag = "📋 Tablo" if doc["is_table_row"] else "📄 Metin"
                    st.markdown(f"**{tag}** — `{doc['topic']}` *(Benzerlik: {score:.3f})*")
                    st.code(doc["text"], language="markdown")

# Yeni Mesaj Girişi
if user_prompt := st.chat_input("Hyprland hakkında bir şey sor veya sohbet et..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        # Arama Adımı
        results, expanded_terms = retrieve(user_prompt, top_k_slider, use_expansion)
        context = "\n\n---\n\n".join([f"[Kaynak: {doc['topic']}]\n{doc['text']}" for doc, _ in results])

        # Sohbet Geçmişi (Hafıza)
        history_text = ""
        for m in st.session_state.messages[-6:-1]:
            role_name = "Kullanıcı" if m["role"] == "user" else "Asistan"
            history_text += f"{role_name}: {m['content']}\n"

        prompt = f"""Sen arkadaş canlısı, samimi ve bilgili bir Hyprland Linux asistanısın.
Kullanıcıyla sohbet ediyorsun. Doğal, akıcı, net ve anlaşılır bir dil kullan.

### Bilgi Bankası (Wiki Kaynakları):
{context}

### Önceki Konuşma Geçmişi:
{history_text}

### Kullanıcının Mesajı:
{user_prompt}

### Talimatlar:
1. Eğer kullanıcı selam veriyor, teşekkür ediyor veya sohbet ediyorsa dostça karşılık ver.
2. Konfigürasyon veya teknik konularda Bilgi Bankası'ndaki verilerden faydalan.
3. Gerekli yerlerde `hyprland.conf` kod blokları paylaş.
"""

        # Stream jeneratörü
        def stream_generator():
            response_stream = client.models.generate_content_stream(
                model=GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temp_slider)
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        try:
            # Cevabı ekrana kelime kelime akıt
            full_response = st.write_stream(stream_generator())

            # Kaynak Detay Menüsü
            with st.expander("🔍 Yanıt İçin Taranan Kaynaklar"):
                if expanded_terms:
                    st.info(f"💡 **Tahmin Edilen Terimler:** {', '.join(expanded_terms)}")
                for doc, score in results:
                    tag = "📋 Tablo Satırı" if doc["is_table_row"] else "📄 Metin Bloğu"
                    st.markdown(f"**{tag}** — `{doc['topic']}` *(Benzerlik: {score:.3f})*")
                    st.code(doc["text"], language="markdown")

            # Hafızaya Kaydet
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": results
            })

        except APIError as e:
            st.error(f"❌ API Hatası: {e}")
        except Exception as e:
            st.error(f"❌ Beklenmeyen Hata: {e}")
