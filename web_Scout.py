




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
    search_topic = "TurboQuant Paper by Google Research"
    print("\n🚀 LAUNCHING TIERED SCOUT...")
    result = app.invoke({"search_query": search_topic, "deep_dive_content": []})
    print("\n🏁 Mission Complete.")


