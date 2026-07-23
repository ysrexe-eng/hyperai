import os

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

st.set_page_config(
    page_title="Hyprland Config Assistant",
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
INTENT_MODEL = "gemma-4-26b-a4b-it"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
INPUT_PATH = "hyprland_dataset.json"
CACHE_PATH = "rag_cache.npz"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 120
SKIP_KEYWORDS = ["readme", "version-selector", "_index", "license"]

API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.title("⚡ Hyprland RAG")
    st.markdown("---")
    
    top_k_slider = st.slider("Documents to Retrieve (Top K)", min_value=3, max_value=20, value=10)
    temp_slider = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    use_expansion = st.checkbox("Query Expansion", value=False, help="Uses LLM to predict additional configuration parameter names.")
    
    st.markdown("---")
    st.markdown("### 📊 Status")
    status_box = st.empty()

def check_intent_with_llm(user_prompt: str, client_obj) -> bool:
    if not client_obj:
        return True
        
    prompt = f"""Analyze the user's message. Does this message require a Hyprland configuration, Linux settings, code snippet, troubleshooting, or technical wiki search?

- If it requires a technical/configuration lookup, reply ONLY with: YES
- If it is a greeting, thank you, casual chat, or follow-up conversation, reply ONLY with: NO

User Message: "{user_prompt}"
Response:"""

    try:
        res = client_obj.models.generate_content(
            model=INTENT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=5
            )
        )
        answer = res.text.strip().upper()
        return "YES" in answer
    except Exception:
        return True

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
                rows.append(f"Setting: {cells[0]}\n" + "\n".join(f"{h}: {v}" for h, v in zip(["Description", "Type", "Default"], cells[1:])))
        return rows

    def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start+size])
            start += size - overlap
        return chunks

    if not os.path.exists(INPUT_PATH):
        st.error(f"❌ File '{INPUT_PATH}' not found!")
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
        st.error(f"❌ Cache file '{CACHE_PATH}' not found!")
        st.stop()

    cache = np.load(CACHE_PATH, allow_pickle=True)
    embeddings = cache["embeddings"]

    return documents, embeddings

if not API_KEY:
    status_box.error("❌ GEMINI_API_KEY not found!")
    st.error("🔑 API Key missing. Please add `GEMINI_API_KEY` to Streamlit Cloud Secrets.")
    st.stop()

try:
    embedder = load_embedder()
    documents, embeddings = load_data_and_cache()
    client = get_gemini_client(API_KEY)
    status_box.success(f"✅ Ready! ({len(embeddings)} embeddings active)")
except Exception as e:
    status_box.error(f"System Error: {e}")
    st.stop()

st.markdown('<div class="main-header">Hyprland Config Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI assistant with intent detection and live response streaming.</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 Sources Used for Response"):
                for doc, score in msg["sources"]:
                    tag = "📋 Table" if doc["is_table_row"] else "📄 Text"
                    st.markdown(f"**{tag}** — `{doc['topic']}` *(Similarity: {score:.3f})*")
                    st.code(doc["text"], language="markdown")

if user_prompt := st.chat_input("Ask a question about Hyprland or start chatting..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        results = []
        expanded_terms = []

        with st.status("⚡ Processing live workflow...", expanded=True) as status:
            status.write(f"🧠 **Intent Analysis:** Evaluating query with `{INTENT_MODEL}`...")
            should_search = check_intent_with_llm(user_prompt, client)

            if should_search:
                status.write("🔍 **Result:** Technical search required. Starting embedding and vector lookup...")

                if use_expansion:
                    status.write("🔍 **Step 1/3:** Predicting configuration terms via LLM...")
                    try:
                        exp_prompt = f"Hyprland config question: '{user_prompt}'. List relevant config parameters in English, comma-separated (e.g., border_size, gaps_in)."
                        res = client.models.generate_content(
                            model=GEN_MODEL, 
                            contents=exp_prompt, 
                            config=types.GenerateContentConfig(temperature=0.2)
                        )
                        expanded_terms = [t.strip() for t in res.text.split(",") if t.strip()]
                        status.write(f"✓ Predicted terms: `{', '.join(expanded_terms)}`")
                    except Exception:
                        status.write("⚠️ Query expansion skipped.")

                status.write("📐 **Step 2/3:** Converting question to vector space (Embedding)...")
                q_text = "query: " + user_prompt + (" " + " ".join(expanded_terms) if expanded_terms else "")
                q_emb = embedder.encode([q_text], convert_to_numpy=True)[0]
                q_emb = q_emb / np.linalg.norm(q_emb)

                status.write("🎯 **Step 3/3:** Searching vector database...")
                scores = embeddings @ q_emb

                keywords = set(re.findall(r'\b[a-zA-Z_]+(?:\.[a-zA-Z_]+)+\b', user_prompt) + re.findall(r'\b[a-zA-Z]+_[a-zA-Z_]+\b', user_prompt) + expanded_terms)
                
                exact_match_idx = []
                if keywords:
                    for i, doc in enumerate(documents):
                        if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', doc["text"].lower()) for kw in keywords if len(kw) > 2):
                            exact_match_idx.append(i)

                exact_match_idx.sort(key=lambda i: scores[i], reverse=True)
                candidate_idx = exact_match_idx + [i for i in np.argsort(scores)[::-1] if i not in set(exact_match_idx)]

                final_idx = []
                for i in candidate_idx:
                    if len(final_idx) >= top_k_slider:
                        break
                    txt = documents[i]["text"]
                    if not any(txt[:100] in documents[j]["text"] for j in final_idx):
                        final_idx.append(i)

                results = [(documents[i], scores[i]) for i in final_idx]
                status.update(label=f"🚀 Found {len(results)} relevant sources. Generating response...", state="complete", expanded=False)
            else:
                status.write("💬 **Result:** Casual chat / follow-up statement detected. Embedding skipped!")
                status.update(label="🚀 Responding directly...", state="complete", expanded=False)

        context = "\n\n---\n\n".join([f"[Source: {doc['topic']}]\n{doc['text']}" for doc, _ in results]) if results else "No search performed (Casual Chat)."

        history_text = ""
        for m in st.session_state.messages[-6:-1]:
            role_name = "User" if m["role"] == "user" else "Assistant"
            history_text += f"{role_name}: {m['content']}\n"

        prompt = f"""You are a friendly, helpful, and knowledgeable Hyprland Linux assistant.
You are chatting with a user. Use a natural, clear, concise, and easy-to-understand tone.

### Knowledge Base (Wiki Sources):
{context}

### Conversation History:
{history_text}

### User's Message:
{user_prompt}

### Instructions:
1. If the user greets, thanks, or engages in casual conversation, respond warmly and naturally.
2. For configuration or technical questions, utilize the Knowledge Base.
3. Provide `hyprland.conf` code blocks whenever relevant.
"""

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
            full_response = st.write_stream(stream_generator())

            if results:
                with st.expander("🔍 Sources Used for Response"):
                    if expanded_terms:
                        st.info(f"💡 **Predicted Terms:** {', '.join(expanded_terms)}")
                    for doc, score in results:
                        tag = "📋 Table Row" if doc["is_table_row"] else "📄 Text Block"
                        st.markdown(f"**{tag}** — `{doc['topic']}` *(Similarity: {score:.3f})*")
                        st.code(doc["text"], language="markdown")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": results
            })

        except APIError as e:
            st.error(f"❌ API Error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {e}")
