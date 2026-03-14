import os
import json
from datetime import datetime
from typing import TypedDict, List
import chromadb
import ollama
from langgraph.graph import StateGraph, START, END
# from duckduckgo_search import DDGS
from ddgs import DDGS

# --- CONFIGURATION ---
LLM_MODEL = "qwen2.5:7b-instruct"
EMBEDDING_MODEL = "nomic-embed-text"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize ChromaDB exactly like digest_generator.py
print("💽 Initializing ChromaDB Vector Store...")
chroma_client = chromadb.PersistentClient(path=os.path.join(SCRIPT_DIR, "morningstar_db"))
collection = chroma_client.get_or_create_collection(name="daily_research")

# --- 1. DEFINE THE STATE (The Graph's Memory) ---
class AgentState(TypedDict):
    search_query: str
    raw_results: List[dict]
    evaluated_results: List[dict]

# --- 2. DEFINE THE NODES (The Agents) ---

# def scout_node(state: AgentState):
#     """AGENT 1: Searches the web for the query."""
#     print(f"🕵️‍♂️ SCOUT: Searching the web for '{state['search_query']}'...")
    
#     # Use DuckDuckGo to search the web free (no API key)
#     results = DDGS().text(state['search_query'], max_results=5)
    
#     raw_results = []
#     for r in results:
#         raw_results.append({
#             "title": r.get('title', 'Unknown'),
#             "url": r.get('href', 'Unknown'),
#             "snippet": r.get('body', '')
#         })
        
#     print(f"🕵️‍♂️ SCOUT: Found {len(raw_results)} raw links.")
#     return {"raw_results": raw_results}

def scout_node(state: AgentState):
    """AGENT 1: Searches the web for the query."""
    print(f"🕵️‍♂️ SCOUT: Searching the web for '{state['search_query']}'...")
    
    # We use the 'html' backend to bypass DuckDuckGo's bot detection
    try:
        results = DDGS().text(state['search_query'], backend="html", max_results=5)
        
        raw_results = []
        # ddgs returns a generator, so we iterate through it safely
        for r in results:
            raw_results.append({
                "title": r.get('title', 'Unknown'),
                "url": r.get('href', 'Unknown'),
                "snippet": r.get('body', '')
            })
            
        print(f"🕵️‍♂️ SCOUT: Found {len(raw_results)} raw links.")
        return {"raw_results": raw_results}
        
    except Exception as e:
        print(f"⚠️ SCOUT ERROR: DuckDuckGo blocked the request. ({e})")
        return {"raw_results": []}

def analyst_node(state: AgentState):
    """AGENT 2: Scores the results using Qwen2.5."""
    print("🧠 ANALYST: Evaluating search results...")
    
    evaluated = []
    system_prompt = """
    You are a strict Data Scientist at Authmind evaluating web search results.
    
    SCORING RUBRIC:
    - 1-3: Completely unrelated or too basic.
    - 4-7: General AI/ML, but not highly specific.
    - 8-10: Highly relevant to Identity Cybersecurity, RAG, or Agentic AI.
    
    Output valid JSON strictly using these exact keys:
    {
      "reasoning": "Explain your thought process in 1 sentence.",
      "score": <integer 1-10 based on the rubric>,
      "summary": "A 1-sentence summary of the finding."
    }
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
            print(f"⚠️ JSON Error on: {item['title'][:30]}")
            
    return {"evaluated_results": evaluated}

def librarian_node(state: AgentState):
    """AGENT 3: Saves high-quality results to ChromaDB."""
    print("📚 LIBRARIAN: Embedding and saving to ChromaDB...")
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    saved_count = 0
    
    for item in state['evaluated_results']:
        document_text = f"Title: {item['title']}\nAbstract: {item['snippet']}\nAI Summary: {item['ai_summary']}"
        
        # Embed using local Nomic model
        embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=document_text)['embedding']
        
        # Save to DB matching the exact schema
        collection.upsert(
            ids=[item['url']], # Using URL as ID
            embeddings=[embedding],
            documents=[document_text],
            metadatas=[{
                "title": item['title'],
                "date_ingested": date_str,
                "score": item['score']
            }]
        )
        saved_count += 1
        
    print(f"✅ LIBRARIAN: Successfully stored {saved_count} new web resources in the database.")
    return state

# --- 3. BUILD THE GRAPH (The State Machine) ---
print("⚙️ Assembling LangGraph State Machine...")
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("scout", scout_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("librarian", librarian_node)

# Define the edges (The Flow)
workflow.add_edge(START, "scout")
workflow.add_edge("scout", "analyst")
workflow.add_edge("analyst", "librarian")
workflow.add_edge("librarian", END)

# Compile the graph
app = workflow.compile()

# --- 4. RUN THE SQUAD ---
if __name__ == "__main__":
    # We can target GitHub specifically using a search operator
    # search_topic = "site:github.com AI identity security agent framework"
    search_topic = "latest open source AI identity security agent frameworks GitHub"
    print("\n🚀 LAUNCHING CREW: Target -> Web Scraping")
    # Execute the graph
    result = app.invoke({"search_query": search_topic})
    print("\n🏁 Mission Complete.")