import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import json
import re
import uuid
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from google.genai.errors import APIError
from ddgs import DDGS

torch.set_num_threads(2)

# Page Configuration
st.set_page_config(
    page_title="HyperAI - Gemma-Powered Agentic RAG",
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
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
INPUT_PATH = "hyprland_dataset.json"
CACHE_PATH = "rag_cache.npz"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 120
SKIP_KEYWORDS = ["readme", "version-selector", "_index", "license"]

# API Keys Initialization
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")

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
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    status_box = st.empty()

# -----------------------------------------------------------------------------
# Core Tools & Gemma Router
# -----------------------------------------------------------------------------

@st.cache_resource
def get_gemini_client(key):
    return genai.Client(api_key=key) if key else None

def scrape_url(url: str, timeout: int = 6) -> str:
    """Scrapes clean text from a web page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for el in soup(["script", "style", "nav", "footer", "header", "aside"]):
                el.extract()
            text = soup.get_text(separator=" ")
            clean_text = "\n".join(chunk.strip() for chunk in text.splitlines() if chunk.strip())
            return clean_text[:4000]
        return f"Error HTTP {response.status_code}"
    except Exception as e:
        return f"Failed to scrape: {e}"

def search_web_multi(queries: list[str], max_results: int = 2) -> tuple[str, list[dict]]:
    """Runs multiple web search queries."""
    docs, metadata, seen = [], [], set()
    ddgs = DDGS()
    for q in queries:
        try:
            for r in ddgs.text(q, max_results=max_results):
                href = r.get('href')
                if href and href not in seen:
                    seen.add(href)
                    docs.append(f"[Web Source: {r.get('title')}]\nQuery: {q}\nSnippet: {r.get('body')}\nURL: {href}")
                    metadata.append({"query": q, "title": r.get('title'), "url": href, "snippet": r.get('body')})
        except Exception:
            continue
    return "\n\n---\n\n".join(docs), metadata

def gemma_agent_router(client_genai, user_prompt: str, context_summary: str, tools_used: list[str]) -> dict:
    """Uses Gemma model as the core Router Reasoning Engine."""
    prompt = f"""You are an Autonomous AI Router Agent powered by Gemma for a Hyprland & Linux System Assistant.
Your task is to analyze the user query and decide what tool to execute next.

User Query: "{user_prompt}"
Tools Executed So Far: {json.dumps(tools_used)}
Current Summary of Gathered Information:
"{context_summary if context_summary else 'No information gathered yet.'}"

STRICT DECISION RULES:
1. GREETINGS & SMALL TALK: If user says casual things ("selam", "hi", "merhaba", "eyvallah"), IMMEDIATELY choose "finish". Do not use any tools!
2. LOCAL RAG: Choose "local_rag" if technical Hyprland/config details are needed and "local_rag" is NOT in Tools Executed.
3. WEB SEARCH: Choose "web_search" ONLY if query needs live info, distros (CachyOS, Arch, etc.), or external tools not found locally.
4. WEB SCRAPE: Choose "web_scrape" ONLY if a URL is explicitly provided in prompt or specific link reading is needed.
5. FINISH: Choose "finish" if gathered info is sufficient to answer completely.

Available actions: "local_rag", "web_search", "web_scrape", "finish"

Output JSON matching schema:
{{
    "next_action": "local_rag" | "web_search" | "web_scrape" | "finish",
    "reasoning": "Clear explanation of why this step was chosen",
    "web_queries": ["query 1 in english", "query 2 in english"],
    "scrape_urls": ["https://..."],
    "expanded_terms": ["term1", "term2"]
}}
"""

    try:
        res = client_genai.models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        return json.loads(res.text)
    except Exception as e:
        return {"next_action": "finish", "reasoning": f"Gemma Router error ({e}). Proceeding to answer."}

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
    st.error("🔑 Please set GEMINI_API_KEY in Streamlit Secrets or Environment.")
    st.stop()

try:
    embedder = load_embedder()
    documents, embeddings = load_data_and_cache()
    client = get_gemini_client(API_KEY)
    status_box.success(f"✅ Gemma Agent Active! ({len(embeddings)} RAG vectors loaded)")
except Exception as e:
    status_box.error(f"Initialization Error: {e}")
    st.stop()

# Main Application UI
st.markdown('<div class="main-header">HyperAI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Gemma-4-31B Powered Autonomous ReAct Agent</div>', unsafe_allow_html=True)

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
if user_prompt := st.chat_input("Ask a question, request a config, or paste a link..."):
    if len(active_chat["messages"]) == 0:
        active_chat["title"] = user_prompt[:25] + ("..." if len(user_prompt) > 25 else "")

    active_chat["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        rag_results = []
        web_context, web_metadata = "", []
        scraped_context, scraped_metadata = "", []
        
        tools_executed = []
        context_accumulator = []

        # Step-by-Step Gemma Reasoning Loop
        with st.status("🧠 Gemma Reasoning Engine thinking...", expanded=True) as status:
            max_steps = 3
            for step in range(1, max_steps + 1):
                summary_str = " | ".join(context_accumulator)[:1000]
                decision = gemma_agent_router(client, user_prompt, summary_str, tools_executed)
                
                action = decision.get("next_action", "finish")
                reasoning = decision.get("reasoning", "")

                status.write(f"💭 **Gemma Step {step}:** *\"{reasoning}\"*")

                if action == "finish":
                    status.write("✅ **Gemma Decision:** Sufficient reasoning completed. Generating response.")
                    break

                # Tool 1: Local Vector RAG
                elif action == "local_rag" and "local_rag" not in tools_executed:
                    tools_executed.append("local_rag")
                    status.write("🔍 **Executing Tool:** Local Vector Dataset Search...")
                    
                    expanded_terms = decision.get("expanded_terms", [])
                    q_text = "query: " + user_prompt + (" " + " ".join(expanded_terms) if expanded_terms else "")
                    q_emb = embedder.encode([q_text], convert_to_numpy=True)[0]
                    q_emb = q_emb / np.linalg.norm(q_emb)

                    scores = embeddings @ q_emb
                    keywords = set(re.findall(r'\b[a-zA-Z_]+\.[a-zA-Z_]+\b', user_prompt) + expanded_terms)
                    
                    exact_match_idx = []
                    if keywords:
                        for i, doc in enumerate(documents):
                            if any(kw.lower() in doc["text"].lower() for kw in keywords):
                                exact_match_idx.append(i)

                    candidate_idx = exact_match_idx + [i for i in np.argsort(scores)[::-1] if i not in set(exact_match_idx)]
                    final_idx = candidate_idx[:top_k_slider]

                    rag_results = [(documents[i], scores[i]) for i in final_idx]
                    status.write(f"   ↳ Retrived {len(rag_results)} local chunks.")
                    context_accumulator.append(f"Local RAG found {len(rag_results)} docs.")

                # Tool 2: Web Search
                elif action == "web_search" and "web_search" not in tools_executed:
                    tools_executed.append("web_search")
                    queries = decision.get("web_queries", [user_prompt])
                    status.write(f"🌐 **Executing Tool:** Dynamic Web Search ({len(queries)} queries)...")
                    for q in queries:
                        status.write(f"   ↳ 🔎 Search: `{q}`")
                    
                    web_context, web_metadata = search_web_multi(queries)
                    status.write(f"   ↳ Found {len(web_metadata)} live web snippets.")
                    context_accumulator.append(f"Web Search found {len(web_metadata)} snippets.")

                # Tool 3: Web Scrape
                elif action == "web_scrape" and "web_scrape" not in tools_executed:
                    tools_executed.append("web_scrape")
                    scrape_urls = decision.get("scrape_urls", [])
                    status.write(f"🕷️ **Executing Tool:** Web Scraping ({len(scrape_urls)} URLs)...")
                    for u in scrape_urls:
                        content = scrape_url(u)
                        if content and not content.startswith("Error"):
                            scraped_context += f"\n[Scraped: {u}]\n{content}\n"
                            scraped_metadata.append({"url": u, "length": len(content), "snippet": content[:200] + "..."})
                    status.write(f"   ↳ Extracted content from {len(scraped_metadata)} pages.")
                    context_accumulator.append(f"Scraped {len(scraped_metadata)} web pages.")

            status.update(label="🚀 Synthesizing final answer...", state="complete", expanded=False)

        # Build Combined Context
        full_context = ""
        if rag_results:
            full_context += "### Local RAG Context:\n" + "\n\n".join([f"[{d['topic']}]\n{d['text']}" for d, _ in rag_results])
        if web_context:
            full_context += "\n\n### Web Search Context:\n" + web_context
        if scraped_context:
            full_context += "\n\n### Scraped Web Content:\n" + scraped_context

        history_text = ""
        for m in active_chat["messages"][-6:-1]:
            role_name = "User" if m["role"] == "user" else "Assistant"
            history_text += f"{role_name}: {m['content']}\n"

        system_prompt = f"""You are HyperAI, an expert system engineer and Hyprland assistant.

Gathered Context:
{full_context if full_context else "No external context used."}

Recent Conversation History:
{history_text}

User Query:
{user_prompt}

Instructions:
1. If user query is a casual greeting, respond naturally without technical jargon.
2. For technical requests, leverage context to provide precise answers and config blocks.
"""

        def stream_generator():
            stream = client.models.generate_content_stream(
                model=GEN_MODEL,
                contents=system_prompt,
                config=types.GenerateContentConfig(temperature=temp_slider)
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text

        try:
            full_response = st.write_stream(stream_generator())

            if rag_results:
                with st.expander("🔍 Used Retrieval Sources"):
                    for doc, score in rag_results:
                        st.markdown(f"**{doc['topic']}** *(Score: {score:.3f})*")
                        st.code(doc["text"], language="markdown")

            if web_metadata:
                with st.expander("🌐 Web Research Sources"):
                    for meta in web_metadata:
                        st.markdown(f"**[{meta['title']}]({meta['url']})** *(Query: `{meta['query']}`)*")

            active_chat["messages"].append({
                "role": "assistant",
                "content": full_response,
                "sources": rag_results,
                "web_sources": web_metadata
            })

        except Exception as e:
            st.error(f"❌ Response Generation Error: {e}")
