




# @@@@@@@@@@@@@@@@@@@@@@ with 2 toggles for two dbs
import streamlit as st
import chromadb
import ollama
import os
from rank_bm25 import BM25Okapi
from ddgs import DDGS # Ensure this is installed: pip install ddgs

# Configuration
LLM_MODEL = "qwen2.5:7b-instruct"
EMBEDDING_MODEL = "nomic-embed-text"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Page Setup
st.set_page_config(page_title="Morningstar AI", page_icon="🌅", layout="wide")
st.title("🌅 Morningstar Research Assistant (Tiered RAG)")

# --- 1. INITIALIZE ENGINES FOR BOTH COLLECTIONS ---
@st.cache_resource
def init_tiered_engines():
    client = chromadb.PersistentClient(path=os.path.join(SCRIPT_DIR, "morningstar_db"))
    
    # Define our two collections
    collections = {
        "Fast Cards": client.get_or_create_collection(name="daily_research"),
        "Deep Dive": client.get_or_create_collection(name="deep_dive_research")
    }
    
    indices = {}
    all_data_map = {}
    
    for name, coll in collections.items():
        data = coll.get(include=['documents', 'metadatas'])
        all_data_map[name] = data
        tokenized = [doc.lower().split() for doc in data['documents']] if data['documents'] else []
        indices[name] = BM25Okapi(tokenized) if tokenized else None
        
    return collections, indices, all_data_map

collections, bm25_indices, all_data_map = init_tiered_engines()

def reciprocal_rank_fusion(dense_ranks, sparse_ranks, k=60):
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_ranks):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ranks):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs]

# --- 2. SIDEBAR WITH TIERED MODE TOGGLE ---
with st.sidebar:
    st.header("⚙️ Search Mode")
    search_mode = st.radio(
        "Select Depth:",
        ["Fast Cards", "Deep Dive"],
        help="Fast Cards = Summaries & News | Deep Dive = Full Technical Content"
    )
    
    st.markdown("---")
    st.header("💽 Database Stats")
    current_data = all_data_map[search_mode]
    st.metric(f"Items in {search_mode}", len(current_data['ids']) if current_data['ids'] else 0)
    st.markdown(f"**Hardware:** RTX 4060 (8GB)\n\n**LLM:** Qwen2.5 (7B)")

# --- 3. ISOLATED MEMORY SETUP ---
# We store messages in a dict keyed by the search_mode to prevent context bleed
if "messages_dict" not in st.session_state:
    st.session_state.messages_dict = {
        "Fast Cards": [],
        "Deep Dive": []
    }

# Display chat history for the CURRENT mode only
for message in st.session_state.messages_dict[search_mode]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. DYNAMIC RETRIEVAL & GENERATION ---
if prompt := st.chat_input("Ask Morningstar..."):
    # Append to the specific mode's history
    st.session_state.messages_dict[search_mode].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Querying {search_mode} Vault..."):
            
            # Step 0: Query Reformulation (using mode-specific history)
            history = st.session_state.messages_dict[search_mode][-5:-1]
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
            
            if history_text.strip():
                reformulate_sys = "Rewrite the latest prompt into a standalone search query. ONLY output the query."
                reformulate_user = f"History:\n{history_text}\n\nLatest Prompt: {prompt}"
                standalone_query = ollama.chat(model=LLM_MODEL, messages=[
                    {'role': 'system', 'content': reformulate_sys},
                    {'role': 'user', 'content': reformulate_user}
                ])['message']['content'].strip()
            else:
                standalone_query = prompt 

            st.caption(f"*(**Internal Search Query:** {standalone_query})*")

            # GET CURRENT CONTEXT
            active_coll = collections[search_mode]
            active_bm25 = bm25_indices[search_mode]
            active_data = all_data_map[search_mode]

            # --- HYBRID RETRIEVAL ---
            final_docs = {'documents': [], 'metadatas': [], 'ids': []}
            
            if active_data['ids']:
                query_embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=standalone_query)['embedding']
                vector_results = active_coll.query(query_embeddings=[query_embedding], n_results=5)
                vector_ids = vector_results['ids'][0]

                tokenized_query = standalone_query.lower().split()
                bm25_scores = active_bm25.get_scores(tokenized_query)
                bm25_ranked = sorted(zip(active_data['ids'], bm25_scores), key=lambda x: x[1], reverse=True)
                keyword_ids = [doc_id for doc_id, score in bm25_ranked[:5] if score > 0]

                final_top_ids = reciprocal_rank_fusion(vector_ids, keyword_ids)[:3]
                final_docs = active_coll.get(ids=final_top_ids)

            # Build Context
            context = ""
            sources = []
            for doc_text, meta, doc_id in zip(final_docs['documents'], final_docs['metadatas'], final_docs['ids']):
                display_title = meta.get('title', 'Untitled')
                context += f"Title: {display_title}\nContent: {doc_text}\n\n"
                score_info = f" (Score: {meta['score']}/10)" if 'score' in meta else ""
                sources.append(f"[{display_title}]({doc_id}){score_info}")

            # --- LLM SYNTHESIS & FALLBACK ---
            system_prompt = f"""
            You are Project Morningstar. You are currently in '{search_mode}' mode.
            Use context to answer. If the answer is not there, say 'I don't have enough data in local memory.'
            CONTEXT:\n{context}
            """
            
            response = ollama.chat(model=LLM_MODEL, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ])
            
            answer = response['message']['content']

            # --- AUTONOMOUS WEB FALLBACK ---
            if "don't have enough data" in answer.lower() or not context:
                st.info("🌐 Local vaults exhausted. Engaging Web Scout Fallback...")
                try:
                    with DDGS() as ddgs:
                        live_results = list(ddgs.text(standalone_query, backend="html", max_results=4))
                        live_context = "\n".join([f"Source: {r['href']}\nSnippet: {r['body']}" for r in live_results])
                    
                    fallback_sys = "You are a live Web Scout. Answer the query using this real-time web data."
                    fallback_resp = ollama.chat(model=LLM_MODEL, messages=[
                        {'role': 'system', 'content': fallback_sys},
                        {'role': 'user', 'content': f"Web Context: {live_context}\n\nUser Query: {prompt}"}
                    ])
                    answer = "🌍 **(Live Web Scout Result):** " + fallback_resp['message']['content']
                    sources = [f"[Web Source]({r['href']})" for r in live_results]
                except Exception as e:
                    answer += f"\n\n*(Fallback failed: {str(e)})*"

            # Display Results
            st.markdown(answer)
            st.markdown("---")
            st.markdown(f"**📚 Sources ({search_mode} Vault):**")
            for source in sources:
                st.markdown(f"- {source}")
                
            st.session_state.messages_dict[search_mode].append({"role": "assistant", "content": answer})