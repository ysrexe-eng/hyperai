import json
import os
import re
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# ---------------------------------------------------------------
# 1. Streamlit Sayfa Yapılandırması
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Hyprland Config Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Hyprland RAG Asistanı")
st.caption("Gelişmiş Tablo Parsing & Semantik Arama Destekli")

# API Key yönetimi (Secrets veya Manuel Girdi)
api_key = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6KQRjeo4QPLMhY_y7EG7e5mdOi5HRwVWgbolRh9mdxW_A")

GEN_MODEL = "gemma-4-31b-it"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
INPUT_PATH = "hyprland_dataset.json"
CACHE_PATH = "rag_cache.npz"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 120
SKIP_KEYWORDS = ["readme", "version-selector", "_index", "license"]

# ---------------------------------------------------------------
# 2. Önbelleğe Alınmış Yüklemeler (Aşırı Hız Sağlar)
# ---------------------------------------------------------------
@st.cache_resource
def get_gemini_client(key):
    return genai.Client(api_key=key)

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_data
def load_dataset_and_embeddings():
    embedder = load_embedder()
    
    # Dokümanları Hazırla
    def parse_markdown_tables(text, topic):
        rows = []
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line.startswith("|") or line.count("|") < 3:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r"[-: ]*", c) for c in cells):
                continue
            if cells and cells[0].lower() in ("name", "key", "setting"):
                continue
            if len(cells) >= 2 and cells[0]:
                row_text = f"Ayar: {cells[0]}\n" + "\n".join(
                    f"{h}: {v}" for h, v in zip(["Açıklama", "Tip", "Varsayılan"], cells[1:])
                )
                rows.append(row_text)
        return rows

    def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
        return chunks

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    documents = []
    for item in raw_data:
        topic = item["instruction"].split(":", 1)[-1].strip()
        if any(kw in topic.lower() for kw in SKIP_KEYWORDS):
            continue

        table_rows = parse_markdown_tables(item["output"], topic)
        for row in table_rows:
            documents.append({"topic": topic, "text": row, "is_table_row": True})

        for chunk in chunk_text(item["output"]):
            if len(chunk.strip()) < 50:
                continue
            documents.append({"topic": topic, "text": chunk, "is_table_row": False})

    # Cache veya Embed İşlemi
    if os.path.exists(CACHE_PATH):
        cache = np.load(CACHE_PATH, allow_pickle=True)
        embeddings = cache["embeddings"]
        cached_texts = cache["texts"]
        if len(cached_texts) != len(documents) or list(cached_texts) != [d["text"] for d in documents]:
            os.remove(CACHE_PATH)
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        chunk_texts = ["passage: " + d["text"] for d in documents]
        embeddings = embedder.encode(chunk_texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        np.savez(CACHE_PATH, embeddings=embeddings, texts=np.array([d["text"] for d in documents], dtype=object))

    return documents, embeddings

# Baştan Yükle
client = get_gemini_client(api_key)
embedder = load_embedder()
documents, embeddings = load_dataset_and_embeddings()

# ---------------------------------------------------------------
# 3. Yardımcı Fonksiyonlar
# ---------------------------------------------------------------
def expand_query(question):
    prompt = f"""Aşağıdaki soru bir Hyprland (Wayland compositor) kullanıcısından geliyor.
Bu soruyla ilgili olabilecek Hyprland config ayar adı/adlarını tahmin et
(örn. border_size, gaps_in, snap.enabled gibi). Sadece olası ayar adlarını
virgülle ayırarak İngilizce yaz, başka açıklama ekleme.

Soru: {question}"""
    try:
        response = client.models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return [t.strip() for t in response.text.split(",") if t.strip()]
    except Exception:
        return []

def extract_keywords(question, expanded_terms=None):
    dotted = re.findall(r'\b[a-zA-Z_]+(?:\.[a-zA-Z_]+)+\b', question)
    underscored = re.findall(r'\b[a-zA-Z]+_[a-zA-Z_]+\b', question)
    keywords = set(dotted + underscored)

    if expanded_terms:
        for term in expanded_terms:
            t_clean = term.strip().strip("`").strip()
            if len(t_clean) > 2:
                keywords.add(t_clean)

    return list(keywords)

def is_near_duplicate(text_a, text_b):
    shorter, longer = sorted([text_a, text_b], key=len)
    if not shorter:
        return False
    return shorter[:200] in longer or (len(shorter) > 30 and shorter[:100] in longer)

def retrieve(question, top_k=10):
    expanded_terms = expand_query(question)
    q_text = "query: " + question + (" " + " ".join(expanded_terms) if expanded_terms else "")
    
    q_emb = embedder.encode([q_text], convert_to_numpy=True)[0]
    q_emb = q_emb / np.linalg.norm(q_emb)
    scores = embeddings @ q_emb

    keywords = extract_keywords(question, expanded_terms)
    exact_match_idx = []
    if keywords:
        for i, doc in enumerate(documents):
            text_lower = doc["text"].lower()
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
                    exact_match_idx.append(i)
                    break

    exact_match_idx.sort(key=lambda i: scores[i], reverse=True)
    exact_set = set(exact_match_idx)
    remaining_idx = [i for i in np.argsort(scores)[::-1] if i not in exact_set]
    candidate_idx = exact_match_idx + remaining_idx

    final_idx = []
    for i in candidate_idx:
        if len(final_idx) >= top_k:
            break
        if not any(is_near_duplicate(documents[i]["text"], documents[j]["text"]) for j in final_idx):
            final_idx.append(i)

    return [(documents[i], scores[i]) for i in final_idx], expanded_terms

# ---------------------------------------------------------------
# 4. Sohbet Geçmişi ve Arayüz Akışı
# ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları çiz
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Kullanılan Kaynaklar"):
                for doc, score in message["sources"]:
                    tag = "📋 Tablo" if doc["is_table_row"] else "📄 Genel"
                    st.write(f"**{tag} - {doc['topic']}** (Skor: {score:.3f})")
                    st.code(doc["text"], language="markdown")

# Yeni Soru
if prompt_text := st.chat_input("Hyprland ayarları hakkında bir soru sorun..."):
    # Kullanıcı mesajı
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Asistan cevabı
    with st.chat_message("assistant"):
        with st.spinner("Wiki taranıyor ve cevap hazırlanıyor..."):
            results, expanded = retrieve(prompt_text)
            
            context_blocks = [f"[Kaynak: {doc['topic']}]\n{doc['text']}" for doc, score in results]
            context = "\n\n---\n\n".join(context_blocks)

            rag_prompt = f"""Aşağıdaki Hyprland wiki kaynaklarını kullanarak kullanıcının sorusunu cevapla.
SADECE verilen kaynaklardaki bilgiyi kullan, uydurma bilgi ekleme. Eğer kaynaklarda
net bir cevap yoksa ama ilgili bir bilgi varsa, onu paylaş ve emin olmadığını belirt.

### Kaynaklar:
{context}

### Soru:
{prompt_text}"""

            response = client.models.generate_content(
                model=GEN_MODEL,
                contents=rag_prompt,
                config=types.GenerateContentConfig(temperature=0.3),
            )
            answer = response.text

            st.markdown(answer)

            # Kaynakları Göster
            with st.expander("📚 Kullanılan Kaynaklar & Detaylar"):
                if expanded:
                    st.write(f"🔍 **Genişletilmiş Anahtar Kelimeler:** `{', '.join(expanded)}`")
                for doc, score in results:
                    tag = "📋 Tablo" if doc["is_table_row"] else "📄 Genel"
                    st.write(f"**{tag} - {doc['topic']}** (Skor: {score:.3f})")
                    st.code(doc["text"], language="markdown")

    # Geçmişe kaydet
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": results
    })
