import os
import re
import json
import uuid
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st
from fastembed import TextEmbedding
from google import genai
from google.genai import types
from ddgs import DDGS

# Streamlit Page Configuration
st.set_page_config(
    page_title="HyperAI - High-Concurrency Agentic RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header { font-size: 2rem; font-weight: 700; color: #58a6ff; margin-bottom: 0.2rem; }
    .sub-header { font-size: 0.9rem; color: #8b949e; margin-bottom: 1.2rem; }
    .cache-badge { background-color: #238636; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# Configuration Constants
GEN_MODEL = "gemma-4-31b-it"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"  # High-performance Multilingual FastEmbed Model
INPUT_PATH = "hyprland_dataset.json"
CACHE_PATH = "rag_cache.npz"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 120
SKIP_KEYWORDS = ["readme", "version-selector", "_index", "license"]

# -----------------------------------------------------------------------------
# 1. Global Thread-Safe API Key Pool (50+ User Capacity)
# -----------------------------------------------------------------------------

class GlobalKeyPool:
    """Thread-safe API key rotator shared across all concurrent sessions."""
    def __init__(self):
        self.lock = threading.Lock()
        self.keys = self._discover_keys()
        self.active_index = 0

    def _discover_keys(self) -> list[str]:
        keys = []
        if "GEMINI_API_KEYS" in st.secrets:
            val = st.secrets["GEMINI_API_KEYS"]
            if isinstance(val, list):
                keys.extend(val)
            elif isinstance(val, str):
                keys.extend([k.strip() for k in val.split(",") if k.strip()])

        for source in [st.secrets, os.environ]:
            for k, v in source.items():
                if (k == "GEMINI_API_KEY" or k.startswith("GEMINI_API_KEY_")) and isinstance(v, str) and v.strip():
                    keys.append(v.strip())

        # Deduplicate keys
        res = []
        for k in keys:
            if k and k not in res:
                res.append(k)
        return res

    def get_client(self) -> genai.Client | None:
        with self.lock:
            if not self.keys:
                return None
            return genai.Client(api_key=self.keys[self.active_index])

    def rotate_on_limit(self) -> bool:
        with self.lock:
            if len(self.keys) <= 1:
                return False
            self.active_index = (self.active_index + 1) % len(self.keys)
            return True

    @property
    def status_info(self) -> str:
        with self.lock:
            if not self.keys:
                return "❌ No API Keys Found"
            return f"🔑 API Key Pool Size: {len(self.keys)} | Active Key Index: #{self.active_index + 1}"

@st.cache_resource
def get_global_key_pool():
    return GlobalKeyPool()

# -----------------------------------------------------------------------------
# 2. Thread-Safe Semantic Cache
# -----------------------------------------------------------------------------

class SemanticCache:
    """In-memory vector cache to answer semantically similar queries in 0 ms."""
    def __init__(self, threshold: float = 0.90):
        self.lock = threading.Lock()
        self.threshold = threshold
        self.cache = []  # List of dicts: {"vector": np.array, "response": str, "sources": list}

    def search(self, query_vector: np.ndarray):
        with self.lock:
            if not self.cache:
                return None, 0.0

            q_norm = query_vector / np.linalg.norm(query_vector)

            for item in self.cache:
                c_norm = item["vector"] / np.linalg.norm(item["vector"])
                similarity = float(np.dot(q_norm, c_norm))

                if similarity >= self.threshold:
                    return item, similarity

            return None, 0.0

    def add(self, query_vector: np.ndarray, response: str, sources: list):
        with self.lock:
            # Prevent unbounded memory usage (max 500 cached entries)
            if len(self.cache) >= 500:
                self.cache.pop(0)

            self.cache.append({
                "vector": query_vector,
                "response": response,
                "sources": sources,
                "timestamp": time.time()
            })

@st.cache_resource
def get_semantic_cache():
    return SemanticCache(threshold=0.90)

# -----------------------------------------------------------------------------
# 3. High-Speed Multilingual FastEmbed & Worker Pool
# -----------------------------------------------------------------------------

@st.cache_resource
def get_fast_embedder():
    # High-performance Multilingual FastEmbed Engine
    return TextEmbedding(model_name=EMBED_MODEL_NAME)

@st.cache_resource
def get_worker_pool():
    # Limit concurrent threads to prevent CPU lockup
    return ThreadPoolExecutor(max_workers=4)

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
        st.error(f"❌ Missing dataset files: '{INPUT_PATH}' or '{CACHE_PATH}'!")
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

# Initialize Infrastructure Components
key_pool = get_global_key_pool()
semantic_cache = get_semantic_cache()
embedder = get_fast_embedder()
worker_pool = get_worker_pool()

try:
    documents, embeddings = load_data_and_cache()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Core Tools & Resilient Agent Router
# -----------------------------------------------------------------------------

def scrape_url(url: str, timeout: int = 5) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for el in soup(["script", "style", "nav", "footer", "header", "aside"]):
                el.extract()
            text = soup.get_text(separator=" ")
            return "\n".join(chunk.strip() for chunk in text.splitlines() if chunk.strip())[:3500]
        return f"HTTP {resp.status_code}"
    except Exception as e:
        return f"Scrape error: {e}"

def search_web_multi(queries: list[str], max_results: int = 2) -> tuple[str, list[dict]]:
    docs, metadata, seen = [], [], set()
    ddgs = DDGS()
    for q in queries:
        try:
            for r in ddgs.text(q, max_results=max_results):
                href = r.get('href')
                if href and href not in seen:
                    seen.add(href)
                    docs.append(f"[Web Source: {r.get('title')}]\nSnippet: {r.get('body')}\nURL: {href}")
                    metadata.append({"query": q, "title": r.get('title'), "url": href, "snippet": r.get('body')})
        except Exception:
            continue
    return "\n\n---\n\n".join(docs), metadata

def gemma_router_resilient(pool: GlobalKeyPool, user_prompt: str, context_summary: str, tools_used: list[str]) -> dict:
    prompt = f"""You are an Autonomous AI Router Agent powered by Gemma for a Hyprland System Assistant.
User Query: "{user_prompt}"
Tools Executed: {json.dumps(tools_used)}
Current Context Summary: "{context_summary}"

RULES:
1. GREETINGS ("hello", "hi", "selam", "merhaba"): Choose "finish" instantly.
2. LOCAL RAG: Choose "local_rag" if Hyprland/Linux configs are needed and "local_rag" NOT in Tools Executed.
3. WEB SEARCH: Choose "web_search" for external distros, live news, or non-Hyprland tools.
4. WEB SCRAPE: Choose "web_scrape" if URL is in user prompt.
5. FINISH: Choose "finish" if info is sufficient.

Return JSON matching schema:
{{
    "next_action": "local_rag" | "web_search" | "web_scrape" | "finish",
    "reasoning": "string",
    "web_queries": ["query1"],
    "scrape_urls": ["url1"],
    "expanded_terms": ["term1"]
}}"""

    for _ in range(max(1, len(pool.keys))):
        client = pool.get_client()
        if not client:
            break
        try:
            res = client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
            )
            return json.loads(res.text)
        except Exception as e:
            err_msg = str(e).lower()
            if any(term in err_msg for term in ["429", "resource_exhausted", "quota", "rate limit"]):
                pool.rotate_on_limit()
                continue
            break
    return {"next_action": "finish", "reasoning": "Direct fallback response."}

# -----------------------------------------------------------------------------
# UI & Session Management
# -----------------------------------------------------------------------------

if "chats" not in st.session_state:
    st.session_state.chats = {}
    default_id = str(uuid.uuid4())
    st.session_state.chats[default_id] = {"title": "New Session", "messages": []}
    st.session_state.active_chat_id = default_id

if "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in st.session_state.chats:
    st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]

with st.sidebar:
    st.title("⚡ HyperAI Agent")
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": f"Chat {len(st.session_state.chats) + 1}", "messages": []}
        st.session_state.active_chat_id = new_id
        st.rerun()

    chat_options = {cid: data["title"] for cid, data in st.session_state.chats.items()}
    chat_ids = list(chat_options.keys())
    curr_idx = chat_ids.index(st.session_state.active_chat_id) if st.session_state.active_chat_id in chat_ids else 0

    selected_chat_id = st.selectbox("Session History", options=chat_ids, format_func=lambda cid: chat_options[cid], index=curr_idx)
    if selected_chat_id != st.session_state.active_chat_id:
        st.session_state.active_chat_id = selected_chat_id
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 System & Server Metrics")
    st.info(key_pool.status_info)
    st.caption(f"🚀 Cached Responses Count: {len(semantic_cache.cache)}")

st.markdown('<div class="main-header">HyperAI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">50+ Concurrency Resilient Agentic RAG Architecture</div>', unsafe_allow_html=True)

active_chat = st.session_state.chats[st.session_state.active_chat_id]

# Message History Render
for msg in active_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("from_cache"):
            st.markdown('<span class="cache-badge">⚡ Cached Response (0 ms)</span>', unsafe_allow_html=True)

# User Input Handling
if user_prompt := st.chat_input("Ask a question, request a config, or paste a URL..."):
    if len(active_chat["messages"]) == 0:
        active_chat["title"] = user_prompt[:25] + "..."

    active_chat["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        # FAST-PATH 1: Vector Generation (FastEmbed - Multilingual ONNX)
        query_vec = list(embedder.embed([f"query: {user_prompt}"]))[0]

        # FAST-PATH 2: Semantic Cache Check (0 ms)
        cached_item, similarity = semantic_cache.search(query_vec)
        if cached_item:
            st.markdown(cached_item["response"])
            st.markdown(f'<span class="cache-badge">⚡ Retrieved from Cache ({similarity*100:.1f}% Similarity)</span>', unsafe_allow_html=True)
            active_chat["messages"].append({
                "role": "assistant",
                "content": cached_item["response"],
                "from_cache": True
            })
            st.stop()

        # FAST-PATH END -> Full Agent Execution
        rag_results = []
        web_context, web_metadata = "", []
        scraped_context = ""
        tools_executed = []
        context_accumulator = []

        with st.status("🧠 Gemma Agent Thinking...", expanded=True) as status:
            for step in range(1, 4):
                summary_str = " | ".join(context_accumulator)[:1000]
                decision = gemma_router_resilient(key_pool, user_prompt, summary_str, tools_executed)
                action = decision.get("next_action", "finish")
                reasoning = decision.get("reasoning", "")

                status.write(f"💭 **Agent Step {step}:** *\"{reasoning}\"*")

                if action == "finish":
                    break

                elif action == "local_rag" and "local_rag" not in tools_executed:
                    tools_executed.append("local_rag")
                    status.write("🔍 **Executing Tool:** Local Vector Dataset Search...")
                    
                    q_norm = query_vec / np.linalg.norm(query_vec)
                    scores = embeddings @ q_norm
                    top_idx = np.argsort(scores)[::-1][:8]
                    rag_results = [(documents[i], scores[i]) for i in top_idx]
                    context_accumulator.append(f"Local RAG found {len(rag_results)} docs.")

                elif action == "web_search" and "web_search" not in tools_executed:
                    tools_executed.append("web_search")
                    queries = decision.get("web_queries", [user_prompt])
                    status.write(f"🌐 **Executing Tool:** Web Search (`{queries[0]}`)...")
                    web_context, web_metadata = search_web_multi(queries)
                    context_accumulator.append(f"Web Search found {len(web_metadata)} snippets.")

                elif action == "web_scrape" and "web_scrape" not in tools_executed:
                    tools_executed.append("web_scrape")
                    scrape_urls = decision.get("scrape_urls", [])
                    if scrape_urls:
                        scraped_context = scrape_url(scrape_urls[0])
                        context_accumulator.append("URL scraped.")

            status.update(label="🚀 Synthesizing Response...", state="complete", expanded=False)

        # Full Context Construction
        full_context = ""
        if rag_results:
            full_context += "### Local RAG Context:\n" + "\n\n".join([f"[{d['topic']}]\n{d['text']}" for d, _ in rag_results])
        if web_context:
            full_context += "\n\n### Web Context:\n" + web_context
        if scraped_context:
            full_context += "\n\n### Scraped Context:\n" + scraped_context

        system_prompt = f"""You are HyperAI, an expert Hyprland & Linux system assistant.
Answer the user's question clearly and concisely.

Context:
{full_context if full_context else "No external context used."}

User Query: {user_prompt}
"""

        # Resilient Multi-Key Streamer
        def stream_response():
            for _ in range(max(1, len(key_pool.keys))):
                client = key_pool.get_client()
                try:
                    stream = client.models.generate_content_stream(
                        model=GEN_MODEL,
                        contents=system_prompt,
                        config=types.GenerateContentConfig(temperature=0.3)
                    )
                    for chunk in stream:
                        if chunk.text:
                            yield chunk.text
                    return
                except Exception as e:
                    err_msg = str(e).lower()
                    if any(term in err_msg for term in ["429", "resource_exhausted", "quota", "rate limit"]):
                        key_pool.rotate_on_limit()
                        continue
                    yield f"❌ Error: {e}"
                    break

        full_response = st.write_stream(stream_response())

        # Cache successful responses for future queries
        if full_response and len(full_response) > 20:
            semantic_cache.add(query_vec, full_response, rag_results)

        active_chat["messages"].append({
            "role": "assistant",
            "content": full_response,
            "from_cache": False
        })
