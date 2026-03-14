# import streamlit as st
# import chromadb
# import ollama
# import os

# # Configuration
# LLM_MODEL = "qwen2.5:7b-instruct"
# EMBEDDING_MODEL = "nomic-embed-text"
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# # Page Setup
# st.set_page_config(page_title="Morningstar AI", page_icon="🌅", layout="wide")
# st.title("🌅 Morningstar Research Assistant")
# st.markdown("Your private, local AI digesting the latest in Identity Cybersecurity & AI.")

# # Initialize DB Connection
# @st.cache_resource
# def get_db_collection():
#     client = chromadb.PersistentClient(path=os.path.join(SCRIPT_DIR, "morningstar_db"))
#     return client.get_or_create_collection(name="daily_research")

# collection = get_db_collection()

# # Sidebar Stats
# with st.sidebar:
#     st.header("💽 Database Stats")
#     try:
#         doc_count = collection.count()
#         st.metric("Papers in Memory", doc_count)
#     except:
#         st.metric("Papers in Memory", 0)
#     st.markdown("---")
#     st.markdown("**Hardware:** RTX 4060 (8GB)\n\n**LLM:** Qwen2.5 (7B)\n\n**Embeddings:** Nomic")

# # Initialize Chat History
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display Chat History
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # User Input
# if prompt := st.chat_input("Ask Morningstar about your research database..."):
#     # Add user message to state and display
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Retrieval & Generation
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
            
#             # --- NEW: STEP 0 - QUERY REFORMULATION ---
#             # Grab the last 4 messages for context (ignoring the one we just appended at [-1])
#             history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:-1]])
            
#             if history_text.strip():
#                 reformulate_sys = "You are a search engine assistant. Read the chat history and the user's latest prompt. Rewrite the latest prompt into a standalone, highly descriptive search query. DO NOT answer the question. ONLY output the rewritten query."
#                 reformulate_user = f"History:\n{history_text}\n\nLatest Prompt: {prompt}"
                
#                 # Ask Qwen to rewrite the query based on conversation context
#                 standalone_query = ollama.chat(model=LLM_MODEL, messages=[
#                     {'role': 'system', 'content': reformulate_sys},
#                     {'role': 'user', 'content': reformulate_user}
#                 ])['message']['content'].strip()
#             else:
#                 # If it's the very first message, just use the raw prompt
#                 standalone_query = prompt 
                
#             # Print the reformulated query to the UI so you can see it working!
#             st.caption(f"*(**Internal Search Query:** {standalone_query})*")

#             # 1. Embed Query (Now using the SMART standalone query)
#             query_embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=standalone_query)['embedding']
            
#             # 2. Retrieve
#             results = collection.query(query_embeddings=[query_embedding], n_results=3)
            
#             if not results['documents'] or not results['documents'][0]:
#                 st.warning("No relevant papers found in the database yet.")
#                 st.stop()
                
#             # 3. Build Context
#             context = ""
#             sources = []
#             for doc, meta, doc_id in zip(results['documents'][0], results['metadatas'][0], results['ids'][0]):
#                 context += f"Title: {meta['title']}\nContent: {doc}\n\n"
#                 sources.append(f"[{meta['title']}]({doc_id}) (Score: {meta['score']}/10)")
                
#             # 4. Prompt LLM for the final answer
#             system_prompt = f"""
#             You are an expert Data Scientist at Authmind. Answer the user's question based ONLY on the provided context.
#             If the answer is not in the context, say "I don't have enough data on this."
#             CONTEXT:\n{context}
#             """
            
#             response = ollama.chat(model=LLM_MODEL, messages=[
#                 {'role': 'system', 'content': system_prompt},
#                 {'role': 'user', 'content': prompt} # Feed it the smart query so it stays on topic
#             ])
            
#             answer = response['message']['content']
            
#             # 5. Display Answer and Sources
#             st.markdown(answer)
#             st.markdown("---")
#             st.markdown("**📚 Sources:**")
#             for source in sources:
#                 st.markdown(f"- {source}")
                
#             # Add assistant response to history
#             st.session_state.messages.append({"role": "assistant", "content": answer})



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ WIthout two ui toggle  fo rtwo collectionS@@@@@@@

# import streamlit as st
# import chromadb
# import ollama
# import os
# from rank_bm25 import BM25Okapi

# # Configuration
# LLM_MODEL = "qwen2.5:7b-instruct"
# EMBEDDING_MODEL = "nomic-embed-text"
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# # Page Setup
# st.set_page_config(page_title="Morningstar AI", page_icon="🌅", layout="wide")
# st.title("🌅 Morningstar Research Assistant (Hybrid RAG)")
# st.markdown("Powered by Vector Embeddings + BM25 Keyword Search + RRF + Memory.")

# # Initialize DB & BM25 Engine
# @st.cache_resource
# def init_search_engines():
#     # 1. Load Vector DB
#     client = chromadb.PersistentClient(path=os.path.join(SCRIPT_DIR, "morningstar_db"))
#     collection = client.get_or_create_collection(name="daily_research")
    
#     # 2. Extract all documents to build the BM25 Keyword Index
#     all_data = collection.get(include=['documents', 'metadatas'])
    
#     # Simple tokenization: lowercase and split by spaces
#     tokenized_corpus = [doc.lower().split() for doc in all_data['documents']] if all_data['documents'] else []
#     bm25_index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    
#     return collection, bm25_index, all_data

# collection, bm25, all_data = init_search_engines()

# def reciprocal_rank_fusion(dense_ranks, sparse_ranks, k=60):
#     """Calculates the RRF score to merge Vector and Keyword results."""
#     rrf_scores = {}
    
#     # Process Vector (Dense) Ranks
#     for rank, doc_id in enumerate(dense_ranks):
#         rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
#     # Process Keyword (Sparse) Ranks
#     for rank, doc_id in enumerate(sparse_ranks):
#         rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
#     # Sort documents by their combined RRF score
#     sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
#     return [doc_id for doc_id, score in sorted_docs]

# # Sidebar Stats
# with st.sidebar:
#     st.header("💽 Database Stats")
#     st.metric("Papers in Memory", len(all_data['ids']) if all_data['ids'] else 0)
#     st.markdown("---")
#     st.markdown("**Architecture:** Hybrid Search (RRF)\n\n**Hardware:** RTX 4060 (8GB)\n\n**LLM:** Qwen2.5 (7B)")

# # Initialize Chat History
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display Chat History
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # User Input
# if prompt := st.chat_input("Ask Morningstar about your research database..."):
#     # Add user message to state and display
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Retrieval & Generation
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking (Memory + Hybrid Search)..."):
            
#             # --- STEP 0: QUERY REFORMULATION (MEMORY) ---
#             # Grab the last 4 messages for context (ignoring the one we just appended at [-1])
#             history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:-1]])
            
#             if history_text.strip():
#                 reformulate_sys = "You are a search engine assistant. Read the chat history and the user's latest prompt. Rewrite the latest prompt into a standalone, highly descriptive search query. DO NOT answer the question. ONLY output the rewritten query."
#                 reformulate_user = f"History:\n{history_text}\n\nLatest Prompt: {prompt}"
                
#                 # Ask Qwen to rewrite the query based on conversation context
#                 standalone_query = ollama.chat(model=LLM_MODEL, messages=[
#                     {'role': 'system', 'content': reformulate_sys},
#                     {'role': 'user', 'content': reformulate_user}
#                 ])['message']['content'].strip()
#             else:
#                 # If it's the very first message, just use the raw prompt
#                 standalone_query = prompt 
                
#             # Print the reformulated query to the UI so you can see it working!
#             st.caption(f"*(**Internal Search Query:** {standalone_query})*")

#             if not all_data['ids']:
#                 st.warning("Database is empty! Run your generator script first.")
#                 st.stop()

#             # --- STEP 1: VECTOR SEARCH (DENSE) ---
#             query_embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=standalone_query)['embedding']
#             vector_results = collection.query(query_embeddings=[query_embedding], n_results=5)
#             vector_ids = vector_results['ids'][0]

#             # --- STEP 2: KEYWORD SEARCH (SPARSE BM25) ---
#             tokenized_query = standalone_query.lower().split()
#             # Get scores for all docs, pair them with IDs, sort, and take top 5
#             bm25_scores = bm25.get_scores(tokenized_query)
#             bm25_ranked = sorted(zip(all_data['ids'], bm25_scores), key=lambda x: x[1], reverse=True)
#             keyword_ids = [doc_id for doc_id, score in bm25_ranked[:5] if score > 0] # Only take >0 matches

#             # --- STEP 3: RECIPROCAL RANK FUSION (RRF MERGE) ---
#             final_top_ids = reciprocal_rank_fusion(vector_ids, keyword_ids)[:3] # Take top 3 combined

#             # Fetch the actual documents using the winning IDs
#             final_docs = collection.get(ids=final_top_ids)

#             # Build Context
#             context = ""
#             sources = []
#             for doc_text, meta, doc_id in zip(final_docs['documents'], final_docs['metadatas'], final_docs['ids']):
#                 context += f"Title: {meta['title']}\nContent: {doc_text}\n\n"
#                 sources.append(f"[{meta['title']}]({doc_id}) (Score: {meta['score']}/10)")
                
#             # --- STEP 4: LLM SYNTHESIS ---
#             # system_prompt = f"""
#             # You are an expert Data Scientist at Authmind. Answer the user's question based ONLY on the provided context.
#             # If the answer is not in the context, say "I don't have enough data on this."
#             # CONTEXT:\n{context}
#             # """
#             system_prompt = f"""
#             You are Project Morningstar, the personal AI Assistant to Varadrajan Kunsavalikar, a Data Scientist at Authmind.
#             Use the provided context to answer the user's query. 
#             If the provided papers are relevant to the user's topic, summarize how they answer the question, even if the exact phrasing differs.
#             If the context is completely unrelated, say "I don't have enough data on this."
#             CONTEXT:\n{context}
#             """
            
#             response = ollama.chat(model=LLM_MODEL, messages=[
#                 {'role': 'system', 'content': system_prompt},
#                 {'role': 'user', 'content': prompt} # Feed it the original conversational prompt
#             ])
            
#             answer = response['message']['content']
            
#             # 5. Display Answer and Sources
#             st.markdown(answer)
#             st.markdown("---")
#             st.markdown("**📚 Sources (Hybrid Retrieved):**")
#             for source in sources:
#                 st.markdown(f"- {source}")
                
#             # Add assistant response to history
#             st.session_state.messages.append({"role": "assistant", "content": answer})




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