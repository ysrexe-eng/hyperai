import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import json
import re
import uuid
import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from google.genai.errors import APIError
from groq import Groq
from ddgs import DDGS

torch.set_num_threads(2)

# Page Configuration
st.set_page_config(
    page_title="HyperAI - Agentic RAG Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header { font-size: 2.2rem; font-weight: 700; color: #58a6ff; margin-bottom: 0.2rem; }
    .sub-header { font-size: 0.95rem; color: #8b949e; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# Configuration Constants
GEN_MODEL = "gemma-4-31b-it"
GROQ_INTENT_MODEL = "llama-3.1-8b-instant"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
INPUT_PATH = "hyprland_dataset.json"
CACHE_PATH = "rag_cache.npz"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 120
SKIP_KEYWORDS = ["readme", "version-selector", "_index", "license"]

# API Keys Initialization
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")

# Session State Initialization
if "chats" not in st.session_state:
    st.session_state.chats = {}
    default_id = str(uuid.uuid4())
    st.session_state.chats[default_id] = {"title": "New Session", "messages": []}
    st.session_state.active_chat_id = default_id

if "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in st.session_state.chats:
    st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]

# Sidebar Interface
with st.sidebar:
    st.title("⚡ HyperAI Agent")
    
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": f"Chat {len(st.session_state.chats) + 1}", "messages": []}
        st.session_state.active_chat_id = new_id
        st.rerun()

    chat_options = {cid: data["title"] for cid, data in st.session_state.chats.items()}
    chat_ids = list(chat_options.keys())
    current_index = chat_ids.index(st.session_state.active_chat_id) if st.session_state.active_chat_id in chat_ids else 0

    selected_chat_id = st.selectbox(
        "Session History",
        options=chat_ids,
        format_func=lambda cid: chat_options[cid],
        index=current_index
    )

    if selected_chat_id != st.session_state.active_chat_id:
        st.session_state.active_chat_id = selected_chat_id
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎛️ Agent Settings")
    top_k_slider = st.slider("Max RAG Documents", min_value=3, max_value=20, value=8)
    temp_slider = st.slider("Model Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    force_web_search = st.checkbox("Force Web Search", value=False, help="Forces the Agent to conduct web research regardless of query intent.")
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    status_box = st.empty()

# -----------------------------------------------------------------------------
# Core Functions & Agent Logic
# -----------------------------------------------------------------------------

def agent_router(user_prompt: str, force_web: bool = False) -> dict:
    """Agent router that analyzes intent and dynamically determines needed tools and queries."""
    if not GROQ_API_KEY:
        return {
            "need_local_rag": True,
            "need_web_search": force_web,
            "web_queries": [user_prompt],
            "expanded_terms": [],
            "reasoning": "Groq API Key missing. Defaulting to standard RAG pipeline."
        }

    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""You are an Autonomous AI Router Agent for a Hyprland & Linux System Assistant.
Analyze the user's input and decide which tools to execute.

User Query: "{user_prompt}"

Task Rules:
1. Determine if local Hyprland documentation search is needed (need_local_rag).
2. Determine if live web search is needed (need_web_search) for Linux distributions (e.g., CachyOS, Arch, Fedora), external tools, recent updates, or general troubleshooting.
3. If web search is needed, generate EXACTLY 2 to 3 distinct, highly focused technical search queries in English to cover different angles.
4. Predict 2-3 specific Hyprland configuration keys or parameters in English (expanded_terms).

Return ONLY a valid JSON object with the following schema:
{{
    "need_local_rag": boolean,
    "need_web_search": boolean,
    "web_queries": ["query 1", "query 2", "query 3"],
    "expanded_terms": ["term1", "term2"],
    "reasoning": "Brief explanation of tool decision"
}}
"""

        res = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_INTENT_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        decision = json.loads(res.choices[0].message.content)
        if force_web:
            decision["need_web_search"] = True
            if not decision.get("web_queries"):
                decision["web_queries"] = [user_prompt, f"{user_prompt} Linux Hyprland config"]
        return decision

    except Exception as e:
        return {
            "need_local_rag": True,
            "need_web_search": force_web,
            "web_queries": [user_prompt, f"{user_prompt} Hyprland"],
            "expanded_terms": [],
            "reasoning": f"Routing failed ({e}). Fallback triggered."
        }

def search_web_multi(queries: list[str], max_results_per_query: int = 2) -> tuple[str, list[dict]]:
    """Executes multiple web searches concurrently and aggregates unique context."""
    aggregated_docs = []
    metadata_list = []
    seen_urls = set()

    ddgs = DDGS()
    for q in queries:
        try:
            results = list(ddgs.text(q, max_results=max_results_per_query))
            for r in results:
                href = r.get('href')
                if href and href not in seen_urls:
                    seen_urls.add(href)
                    doc_str = f"[Web Source: {r.get('title')}]\nQuery: {q}\nSnippet: {r.get('body')}\nURL: {href}"
                    aggregated_docs.append(doc_str)
                    metadata_list.append({"query": q, "title": r.get('title'), "url": href, "snippet": r.get('body')})
        except Exception:
            continue

    formatted_context = "\n\n---\n\n".join(aggregated_docs) if aggregated_docs else ""
    return formatted_context, metadata_list

@st.cache_resource
def get_gemini_client(key):
    return genai.Client(api_key=key) if key else None

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

    if not os.path.exists(INPUT_PATH) or not os.path.exists(CACHE_PATH):
        st.error(f"❌ Missing dataset files ('{INPUT_PATH}' or '{CACHE_PATH}')!")
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

    cache = np.load(CACHE_PATH, allow_pickle=True)
    embeddings = cache["embeddings"]

    return documents, embeddings

# Initialize Backend
if not API_KEY:
    status_box.error("❌ GEMINI_API_KEY missing!")
    st.error("🔑 Please set GEMINI_API_KEY in your environment or Streamlit Secrets.")
    st.stop()

try:
    embedder = load_embedder()
    documents, embeddings = load_data_and_cache()
    client = get_gemini_client(API_KEY)
    status_box.success(f"✅ Ready! ({len(embeddings)} RAG vectors active)")
except Exception as e:
    status_box.error(f"Initialization Error: {e}")
    st.stop()

# Main Application UI
st.markdown('<div class="main-header">HyperAI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autonomous RAG & Multi-Query Search Assistant for Hyprland & Linux Systems</div>', unsafe_allow_html=True)

active_chat = st.session_state.chats[st.session_state.active_chat_id]

# Render Chat History
for msg in active_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 Used Retrieval Sources"):
                for doc, score in msg["sources"]:
                    tag = "📋 Table" if doc["is_table_row"] else "📄 Text"
                    st.markdown(f"**{tag}** — `{doc['topic']}` *(Similarity: {score:.3f})*")
                    st.code(doc["text"], language="markdown")
        if "web_sources" in msg and msg["web_sources"]:
            with st.expander("🌐 Web Research Sources"):
                for meta in msg["web_sources"]:
                    st.markdown(f"**[{meta['title']}]({meta['url']})** *(Query: `{meta['query']}`)*")
                    st.caption(meta['snippet'])

# Chat Input Handler
if user_prompt := st.chat_input("Ask about Hyprland, Linux distros, or request configurations..."):
    if len(active_chat["messages"]) == 0:
        active_chat["title"] = user_prompt[:25] + ("..." if len(user_prompt) > 25 else "")

    active_chat["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        results = []
        web_context = ""
        web_metadata = []

        with st.status("🤖 Agent executing workflow...", expanded=True) as status:
            # Step 1: Agent Decision Making
            status.write("🧠 **Agent Router:** Analyzing intent and planning tool strategy...")
            decision = agent_router(user_prompt, force_web=force_web_search)
            status.write(f"💭 **Agent Reasoning:** *\"{decision.get('reasoning')}\"*")

            # Step 2: Local RAG Retrieval (if requested)
            if decision.get("need_local_rag", True):
                status.write("🔍 **Tool Execution:** Searching local vector dataset...")
                expanded_terms = decision.get("expanded_terms", [])
                
                if expanded_terms:
                    status.write(f"💡 **Predicted Config Keys:** `{', '.join(expanded_terms)}`")

                q_text = "query: " + user_prompt + (" " + " ".join(expanded_terms) if expanded_terms else "")
                q_emb = embedder.encode([q_text], convert_to_numpy=True)[0]
                q_emb = q_emb / np.linalg.norm(q_emb)

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
                status.write(f"✓ Retrieved **{len(results)}** local documentation chunks.")

            # Step 3: Dynamic Multi-Query Web Search (if requested)
            if decision.get("need_web_search", False):
                queries = decision.get("web_queries", [user_prompt])
                status.write(f"🌐 **Tool Execution:** Initiating {len(queries)} dynamic web searches...")
                
                for idx, q in enumerate(queries, 1):
                    status.write(f"   ↳ 🔎 Search Query {idx}/{len(queries)}: `{q}`")
                
                web_context, web_metadata = search_web_multi(queries, max_results_per_query=2)
                status.write(f"✓ Collected **{len(web_metadata)}** unique live web snippets.")

            status.update(label="🚀 Synthesizing response...", state="complete", expanded=False)

        # Step 4: Build Context and Stream Gemini Response
        local_context = "\n\n---\n\n".join([f"[Source: {doc['topic']}]\n{doc['text']}" for doc, _ in results]) if results else "No local wiki context used."
        
        full_context = f"### Local Knowledge Base:\n{local_context}"
        if web_context:
            full_context += f"\n\n### Web Search Context:\n{web_context}"

        history_text = ""
        for m in active_chat["messages"][-6:-1]:
            role_name = "User" if m["role"] == "user" else "Assistant"
            history_text += f"{role_name}: {m['content']}\n"

        system_prompt = f"""You are HyperAI, an expert system engineer and Hyprland assistant.
You are interacting with the user. Use a clear, concise, well-formatted, and helpful tone.

{full_context}

### Recent Conversation History:
{history_text}

### Current User Query:
{user_prompt}

### Instructions:
1. Respond naturally in the language used by the user (or English if technical standard).
2. Use the provided Knowledge Base and Web Search Context to give accurate answers.
3. Provide code blocks (e.g., `hyprland.conf`, bash scripts) whenever technical configuration is requested.
"""

        def stream_generator():
            response_stream = client.models.generate_content_stream(
                model=GEN_MODEL,
                contents=system_prompt,
                config=types.GenerateContentConfig(temperature=temp_slider)
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        try:
            full_response = st.write_stream(stream_generator())

            # Source Render Expanders
            if results:
                with st.expander("🔍 Used Retrieval Sources"):
                    for doc, score in results:
                        tag = "📋 Table Row" if doc["is_table_row"] else "📄 Text Block"
                        st.markdown(f"**{tag}** — `{doc['topic']}` *(Similarity: {score:.3f})*")
                        st.code(doc["text"], language="markdown")

            if web_metadata:
                with st.expander("🌐 Web Research Sources"):
                    for meta in web_metadata:
                        st.markdown(f"**[{meta['title']}]({meta['url']})** *(Query: `{meta['query']}`)*")
                        st.caption(meta['snippet'])

            active_chat["messages"].append({
                "role": "assistant",
                "content": full_response,
                "sources": results,
                "web_sources": web_metadata
            })

        except APIError as e:
            st.error(f"❌ Gemini API Error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {e}")
