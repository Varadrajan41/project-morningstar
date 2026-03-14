import arxiv
import ollama
import json
import os
import chromadb
from datetime import datetime

# Configuration
LLM_MODEL = "qwen2.5:7b-instruct"
EMBEDDING_MODEL = "nomic-embed-text"
MAX_RESULTS = 20
SEARCH_QUERY = "cat:cs.AI OR cat:cs.CR"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize ChromaDB Persistent Client
print("💽 Initializing ChromaDB Vector Store...")
chroma_client = chromadb.PersistentClient(path=os.path.join(SCRIPT_DIR, "morningstar_db"))
collection = chroma_client.get_or_create_collection(name="daily_research")

def fetch_latest_papers():
    print(f"🔍 Fetching top {MAX_RESULTS} papers from ArXiv...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    return list(client.results(search))

def analyze_paper_with_ollama(title, abstract):
    print(f"🧠 Analyzing & Extracting JSON: {title[:50]}...")
    
    # 1. Added a strict Scoring Rubric
    # 2. Added a "reasoning" key so it thinks BEFORE it scores
    system_prompt = """
    You are a strict Data Scientist at Authmind evaluating research papers.
    
    SCORING RUBRIC:
    - 1-3: Completely unrelated to Cybersecurity, RAG, or Agents (e.g., color theory, physics, biology).
    - 4-7: General AI/ML, but not specifically Identity Security or Agentic AI.
    - 8-10: Highly relevant to Identity Cybersecurity, RAG, or Agentic AI.
    
    Output valid JSON strictly using these exact keys:
    {
      "reasoning": "Explain your thought process for the score in 1 sentence.",
      "score": <integer 1-10 based on the rubric>,
      "summary": "A 2-sentence summary of the paper.",
      "use_case": "One practical industry use case."
    }
    """
    prompt = f"Title: {title}\nAbstract: {abstract}"
    
    # 3. Added options={'temperature': 0.1} for strict, logical output
    response = ollama.chat(model=LLM_MODEL, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ], format='json', options={'temperature': 0.1})
    
    try:
        return json.loads(response['message']['content'])
    except json.JSONDecodeError:
        return {"summary": "Error parsing summary.", "score": 0, "use_case": "N/A", "reasoning": "Error"}

# def analyze_paper_with_ollama(title, abstract):
#     print(f"🧠 Analyzing & Extracting JSON: {title[:50]}...")
#     system_prompt = """
#     You are an expert Data Scientist at Authmind. Analyze research for Identity Cybersecurity, RAG, and Agentic AI.
#     Output valid JSON strictly using these keys:
#     {
#       "summary": "A 2-sentence summary of the paper.",
#       "score": <integer 1-10>,
#       "use_case": "One practical industry use case."
#     }
#     """
#     prompt = f"Title: {title}\nAbstract: {abstract}"
    
#     response = ollama.chat(model=LLM_MODEL, messages=[
#         {'role': 'system', 'content': system_prompt},
#         {'role': 'user', 'content': prompt}
#     ], format='json')
    
#     try:
#         return json.loads(response['message']['content'])
#     except json.JSONDecodeError:
#         return {"summary": "Error parsing summary.", "score": 0, "use_case": "N/A"}

def generate_embeddings(text):
    # Use Ollama to generate embeddings locally
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response['embedding']

# def process_and_store_papers(papers):
#     date_str = datetime.now().strftime("%Y-%m-%d")
#     filename = os.path.join(SCRIPT_DIR, f"Morningstar_Digest_{date_str}.md")
    
#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(f"# 🌅 Project Morningstar: Daily AI & Cyber Digest\n")
#         f.write(f"**Date:** {date_str}\n\n---\n\n")
        
#         for paper in papers:
#             # 1. Get the structured JSON analysis
#             analysis = analyze_paper_with_ollama(paper.title, paper.summary)
            
#             # 2. Write to Markdown (The Human UI)
#             f.write(f"## [{paper.title}]({paper.entry_id})\n")
#             f.write(f"**Relevance Score:** {analysis.get('score', 0)}/10\n\n")
#             f.write(f"**Summary:** {analysis.get('summary', '')}\n\n")
#             f.write(f"**Authmind Use Case:** {analysis.get('use_case', '')}\n\n")
#             f.write("---\n")
            
#             # 3. Store in Vector Database (The Agent UI)
#             print(f"💾 Saving to Vector DB: {paper.entry_id}")
#             # We embed the original abstract + our AI's summary for rich semantic search
#             document_text = f"Title: {paper.title}\nAbstract: {paper.summary}\nAI Summary: {analysis.get('summary', '')}"
#             embedding = generate_embeddings(document_text)
            
#             collection.upsert(
#                 ids=[paper.entry_id], # Using the ArXiv URL as a unique ID prevents duplicates
#                 embeddings=[embedding],
#                 documents=[document_text],
#                 metadatas=[{
#                     "title": paper.title,
#                     "date_ingested": date_str,
#                     "score": analysis.get('score', 0),
#                     "use_case": analysis.get('use_case', '')
#                 }]
#             )
            
#     print(f"✅ Digest saved to {filename}")
#     print(f"✅ Papers successfully embedded into ChromaDB!")

# Change this at the top of your file
MAX_RESULTS = 20 

def process_and_store_papers(papers):
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(SCRIPT_DIR, f"Morningstar_Digest_{date_str}.md")
    
    saved_count = 0
    rejected_count = 0
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 🌅 Project Morningstar: Daily AI & Cyber Digest\n")
        f.write(f"**Date:** {date_str}\n\n---\n\n")
        
        for paper in papers:
            analysis = analyze_paper_with_ollama(paper.title, paper.summary)
            score = analysis.get('score', 0)
            
            # THE QUALITY CONTROL GATE
            if score >= 7:
                saved_count += 1
                f.write(f"## [{paper.title}]({paper.entry_id})\n")
                f.write(f"**Relevance Score:** {score}/10\n")
                f.write(f"**AI Reasoning:** {analysis.get('reasoning', '')}\n\n")
                f.write(f"**Summary:** {analysis.get('summary', '')}\n\n")
                f.write(f"**Authmind Use Case:** {analysis.get('use_case', '')}\n\n")
                f.write("---\n")
                
                print(f"⭐ High Score ({score}/10)! Saving to Vector DB: {paper.title[:30]}...")
                document_text = f"Title: {paper.title}\nAbstract: {paper.summary}\nAI Summary: {analysis.get('summary', '')}"
                embedding = generate_embeddings(document_text)
                
                collection.upsert(
                    ids=[paper.entry_id],
                    embeddings=[embedding],
                    documents=[document_text],
                    metadatas=[{
                        "title": paper.title,
                        "date_ingested": date_str,
                        "score": score
                    }]
                )
            else:
                rejected_count += 1
                print(f"🗑️ Rejected ({score}/10): {paper.title[:30]}...")
                
    print(f"\n✅ Ingestion Complete! Saved {saved_count} papers. Filtered out {rejected_count} irrelevant papers.")

if __name__ == "__main__":
    print("🚀 Starting Project Morningstar (V3 Vector Memory)...")
    papers = fetch_latest_papers()
    process_and_store_papers(papers)