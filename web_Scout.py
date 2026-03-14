# import os
# import json
# from datetime import datetime
# from typing import TypedDict, List
# import chromadb
# import ollama
# from langgraph.graph import StateGraph, START, END
# # from duckduckgo_search import DDGS
# from ddgs import DDGS

# # --- CONFIGURATION ---
# LLM_MODEL = "qwen2.5:7b-instruct"
# EMBEDDING_MODEL = "nomic-embed-text"
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# # Initialize ChromaDB exactly like digest_generator.py
# print("💽 Initializing ChromaDB Vector Store...")
# chroma_client = chromadb.PersistentClient(path=os.path.join(SCRIPT_DIR, "morningstar_db"))
# collection = chroma_client.get_or_create_collection(name="daily_research")

# # --- 1. DEFINE THE STATE (The Graph's Memory) ---
# class AgentState(TypedDict):
#     search_query: str
#     raw_results: List[dict]
#     evaluated_results: List[dict]

# # --- 2. DEFINE THE NODES (The Agents) ---

# # def scout_node(state: AgentState):
# #     """AGENT 1: Searches the web for the query."""
# #     print(f"🕵️‍♂️ SCOUT: Searching the web for '{state['search_query']}'...")
    
# #     # Use DuckDuckGo to search the web free (no API key)
# #     results = DDGS().text(state['search_query'], max_results=5)
    
# #     raw_results = []
# #     for r in results:
# #         raw_results.append({
# #             "title": r.get('title', 'Unknown'),
# #             "url": r.get('href', 'Unknown'),
# #             "snippet": r.get('body', '')
# #         })
        
# #     print(f"🕵️‍♂️ SCOUT: Found {len(raw_results)} raw links.")
# #     return {"raw_results": raw_results}

# # Query Rewrite Scout
# # def scout_node(state: AgentState):
# #     """AGENT 1: The Intelligent Scout - Rewrites query for variety."""
# #     print(f"🕵️‍♂️ SCOUT: Original Intent -> '{state['search_query']}'")
    
# #     # NEW: Tell Qwen to suggest 3 different high-quality searches
# #     rewrite_prompt = f"Given this research topic: '{state['search_query']}', suggest 3 distinct search queries to get diverse results from GitHub, HuggingFace, and Security Blogs. Output ONLY the 3 queries, one per line."
# #     response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': rewrite_prompt}])
# #     queries = response['message']['content'].strip().split('\n')
    
# #     all_raw_results = []
# #     with DDGS() as ddgs:
# #         for q in queries[:3]: # We run the 3 smarter queries
# #             print(f"🔍 Executing Smart Search: {q.strip()}")
# #             try:
# #                 # Backend 'html' is safer, max_results=3 per query = 9 total
# #                 results = ddgs.text(q.strip(), backend="html", max_results=3)
# #                 for r in results:
# #                     all_raw_results.append({
# #                         "title": r.get('title', 'Unknown'),
# #                         "url": r.get('href', 'Unknown'),
# #                         "snippet": r.get('body', '')
# #                     })
# #             except Exception as e:
# #                 print(f"⚠️ Search failed for '{q}': {e}")

# #     print(f"🕵️‍♂️ SCOUT: Found {len(all_raw_results)} total links using Smart Queries.")
# #     return {"raw_results": all_raw_results}

# def scout_node(state: AgentState):
#     """AGENT 1: Searches the web for the query."""
#     print(f"🕵️‍♂️ SCOUT: Searching the web for '{state['search_query']}'...")
    
#     # We use the 'html' backend to bypass DuckDuckGo's bot detection
#     try:
#         results = DDGS().text(state['search_query'], backend="html", max_results=5)
        
#         raw_results = []
#         # ddgs returns a generator, so we iterate through it safely
#         for r in results:
#             raw_results.append({
#                 "title": r.get('title', 'Unknown'),
#                 "url": r.get('href', 'Unknown'),
#                 "snippet": r.get('body', '')
#             })
            
#         print(f"🕵️‍♂️ SCOUT: Found {len(raw_results)} raw links.")
#         return {"raw_results": raw_results}
        
#     except Exception as e:
#         print(f"⚠️ SCOUT ERROR: DuckDuckGo blocked the request. ({e})")
#         return {"raw_results": []}

# def analyst_node(state: AgentState):
#     """AGENT 2: Scores the results using Qwen2.5."""
#     print("🧠 ANALYST: Evaluating search results...")
    
#     evaluated = []
#     system_prompt = """
#     You are a strict Data Scientist at Authmind evaluating web search results.
    
#     SCORING RUBRIC:
#     - 1-3: Completely unrelated or too basic.
#     - 4-7: General AI/ML, but not highly specific.
#     - 8-10: Highly relevant to Identity Cybersecurity, RAG, or Agentic AI.
    
#     Output valid JSON strictly using these exact keys:
#     {
#       "reasoning": "Explain your thought process in 1 sentence.",
#       "score": <integer 1-10 based on the rubric>,
#       "summary": "A 1-sentence summary of the finding."
#     }
#     """
    
#     for item in state['raw_results']:
#         prompt = f"Title: {item['title']}\nSnippet: {item['snippet']}"
        
#         response = ollama.chat(model=LLM_MODEL, messages=[
#             {'role': 'system', 'content': system_prompt},
#             {'role': 'user', 'content': prompt}
#         ], format='json', options={'temperature': 0.1})
        
#         try:
#             analysis = json.loads(response['message']['content'])
#             score = analysis.get('score', 0)
            
#             if score >= 7:
#                 print(f"⭐ High Score ({score}/10) Approved: {item['title'][:30]}...")
#                 item['score'] = score
#                 item['ai_summary'] = analysis.get('summary', '')
#                 evaluated.append(item)
#             else:
#                 print(f"🗑️ Rejected ({score}/10): {item['title'][:30]}...")
                
#         except json.JSONDecodeError:
#             print(f"⚠️ JSON Error on: {item['title'][:30]}")
            
#     return {"evaluated_results": evaluated}

# def librarian_node(state: AgentState):
#     """AGENT 3: Saves high-quality results to ChromaDB."""
#     print("📚 LIBRARIAN: Embedding and saving to ChromaDB...")
    
#     date_str = datetime.now().strftime("%Y-%m-%d")
#     saved_count = 0
    
#     for item in state['evaluated_results']:
#         document_text = f"Title: {item['title']}\nAbstract: {item['snippet']}\nAI Summary: {item['ai_summary']}"
        
#         # Embed using local Nomic model
#         embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=document_text)['embedding']
        
#         # Save to DB matching the exact schema
#         collection.upsert(
#             ids=[item['url']], # Using URL as ID
#             embeddings=[embedding],
#             documents=[document_text],
#             metadatas=[{
#                 "title": item['title'],
#                 "date_ingested": date_str,
#                 "score": item['score']
#             }]
#         )
#         saved_count += 1
        
#     print(f"✅ LIBRARIAN: Successfully stored {saved_count} new web resources in the database.")
#     return state

# # --- 3. BUILD THE GRAPH (The State Machine) ---
# print("⚙️ Assembling LangGraph State Machine...")
# workflow = StateGraph(AgentState)

# # Add the nodes
# workflow.add_node("scout", scout_node)
# workflow.add_node("analyst", analyst_node)
# workflow.add_node("librarian", librarian_node)

# # Define the edges (The Flow)
# workflow.add_edge(START, "scout")
# workflow.add_edge("scout", "analyst")
# workflow.add_edge("analyst", "librarian")
# workflow.add_edge("librarian", END)

# # Compile the graph
# app = workflow.compile()

# # --- 4. RUN THE SQUAD ---
# if __name__ == "__main__":
#     # We can target GitHub specifically using a search operator
#     # search_topic = "site:github.com AI identity security agent framework"
#     search_topic = "Cisco Foundation-sec-8b open source security model research paper huggingface"
#     print("\n🚀 LAUNCHING CREW: Target -> Web Scraping")
#     # Execute the graph
#     result = app.invoke({"search_query": search_topic})
#     print("\n🏁 Mission Complete.")



# # if __name__ == "__main__":
# #     # We define a list of targets to ensure we get a diverse 'brain'
# #     topics = [
# #         "latest identity security models site:huggingface.co",
# #         "agentic ai security frameworks site:github.com",
# #         "LLM prompt injection vulnerabilities 2026"
# #     ]
    
# #     print("\n🚀 LAUNCHING MULTI-SOURCE SCOUT...")
# #     for topic in topics:
# #         print(f"\n--- Target: {topic} ---")
# #         app.invoke({"search_query": topic})
        
# #     print("\n🏁 All missions complete. Your ChromaDB is now enriched.")




# NEW CODE = added deep reasearcher
import os
import json
from datetime import datetime
from typing import TypedDict, List
import chromadb
import ollama
from langgraph.graph import StateGraph, START, END
from ddgs import DDGS
import trafilatura  # <-- ADDED for Deep Research

# --- CONFIGURATION ---
LLM_MODEL = "qwen2.5:7b-instruct"
EMBEDDING_MODEL = "nomic-embed-text"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize ChromaDB
print("💽 Initializing ChromaDB Tiered Storage...")
chroma_client = chromadb.PersistentClient(path=os.path.join(SCRIPT_DIR, "morningstar_db"))

# COLLECTION 1: The "Fast Cards" (ArXiv + Web Snippets)
collection = chroma_client.get_or_create_collection(name="daily_research")

# COLLECTION 2: The "Deep Vault" (Full Website/Paper Text) - NEW!
deep_collection = chroma_client.get_or_create_collection(name="deep_dive_research")

# --- 1. DEFINE THE STATE ---
class AgentState(TypedDict):
    search_query: str
    raw_results: List[dict]
    evaluated_results: List[dict]
    # NEW: Store full text content for high-scoring hits
    deep_dive_content: List[dict]

# --- 2. DEFINE THE NODES ---

def scout_node(state: AgentState):
    """AGENT 1: Searches the web (Original Logic Kept)."""
    print(f"🕵️‍♂️ SCOUT: Searching the web for '{state['search_query']}'...")
    try:
        results = DDGS().text(state['search_query'], backend="html", max_results=5)
        raw_results = []
        for r in results:
            raw_results.append({
                "title": r.get('title', 'Unknown'),
                "url": r.get('href', 'Unknown'),
                "snippet": r.get('body', '')
            })
        print(f"🕵️‍♂️ SCOUT: Found {len(raw_results)} raw links.")
        return {"raw_results": raw_results}
    except Exception as e:
        print(f"⚠️ SCOUT ERROR: {e}")
        return {"raw_results": []}

def analyst_node(state: AgentState):
    """AGENT 2: Scores results (Original Logic Kept)."""
    print("🧠 ANALYST: Evaluating search results...")
    evaluated = []
    system_prompt = """
    You are a strict Data Scientist at Authmind. 
    Rubric: 8-10 for Identity Cybersecurity/Agentic AI.
    Output JSON: {"reasoning": "...", "score": int, "summary": "..."}
    """
    for item in state['raw_results']:
        prompt = f"Title: {item['title']}\nSnippet: {item['snippet']}"
        response = ollama.chat(model=LLM_MODEL, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ], format='json', options={'temperature': 0.1})
        try:
            analysis = json.loads(response['message']['content'])
            score = analysis.get('score', 0)
            if score >= 7:
                print(f"⭐ High Score ({score}/10) Approved: {item['title'][:30]}...")
                item['score'] = score
                item['ai_summary'] = analysis.get('summary', '')
                evaluated.append(item)
            else:
                print(f"🗑️ Rejected ({score}/10): {item['title'][:30]}...")
        except json.JSONDecodeError:
            print(f"⚠️ JSON Error")
    return {"evaluated_results": evaluated}

# --- NEW NODE: THE RESEARCHER ---
def researcher_node(state: AgentState):
    """AGENT 3: Visits high-score links and extracts FULL text."""
    print("🔍 RESEARCHER: Clicking links for deep dive (Score 9+ only)...")
    deep_content = []
    
    for item in state['evaluated_results']:
        # We only do full-text extraction for top-tier content (9 or 10)
        # This keeps the database from becoming "noisy" with mid-tier fluff.
        if item['score'] >= 9:
            print(f"📖 Reading full page: {item['title'][:30]}...")
            downloaded = trafilatura.fetch_url(item['url'])
            if downloaded:
                full_text = trafilatura.extract(downloaded)
                if full_text:
                    deep_content.append({
                        "url": item['url'],
                        "full_text": full_text,
                        "title": item['title']
                    })
    return {"deep_dive_content": deep_content}

def librarian_node(state: AgentState):
    """AGENT 4: Routes data to appropriate collections (Updated)."""
    print("📚 LIBRARIAN: Updating Tiered Memory...")
    date_str = datetime.now().strftime("%Y-%m-%d")

    # 1. SAVE SNIPPETS TO DAILY_RESEARCH (For the "Fast Cards" UI)
    for item in state['evaluated_results']:
        doc_text = f"Title: {item['title']}\nAbstract: {item['snippet']}\nAI Summary: {item['ai_summary']}"
        embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=doc_text)['embedding']
        collection.upsert(
            ids=[item['url']], 
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{"title": item['title'], "date_ingested": date_str, "score": item['score'], "type": "snippet"}]
        )

    # 2. SAVE FULL TEXT TO DEEP_DIVE_RESEARCH (For technical specifics)
    # This is where the Cisco 8B paper details will live!
    deep_count = 0
    if state.get('deep_dive_content'):
        for deep_item in state['deep_dive_content']:
            # Chunking the full text briefly (simplistic approach for now)
            # We save the first 4000 chars to avoid hitting embedding limits
            truncated_text = deep_item['full_text'][:4000] 
            embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=truncated_text)['embedding']
            deep_collection.upsert(
                ids=[f"full_{deep_item['url']}"],
                embeddings=[embedding],
                documents=[truncated_text],
                metadatas=[{"title": deep_item['title'], "date_ingested": date_str, "type": "full_text"}]
            )
            deep_count += 1
            
    print(f"✅ LIBRARIAN: Stored {len(state['evaluated_results'])} snippets and {deep_count} full-text deep dives.")
    return state

# --- 3. BUILD THE GRAPH (Updated Flow) ---
print("⚙️ Assembling Tiered LangGraph...")
workflow = StateGraph(AgentState)

workflow.add_node("scout", scout_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("researcher", researcher_node) # <-- ADDED
workflow.add_node("librarian", librarian_node)

workflow.add_edge(START, "scout")
workflow.add_edge("scout", "analyst")
workflow.add_edge("analyst", "researcher") # <-- ROUTE THROUGH RESEARCHER
workflow.add_edge("researcher", "librarian")
workflow.add_edge("librarian", END)

app = workflow.compile()

# --- 4. RUN THE SQUAD ---
if __name__ == "__main__":
    search_topic = "Cisco Foundation-sec-8b open source security model research paper huggingface"
    print("\n🚀 LAUNCHING TIERED SCOUT...")
    result = app.invoke({"search_query": search_topic, "deep_dive_content": []})
    print("\n🏁 Mission Complete.")


