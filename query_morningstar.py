import os
import chromadb
import ollama

# Configuration
LLM_MODEL = "qwen2.5:7b-instruct"
EMBEDDING_MODEL = "nomic-embed-text"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Connect to the local vector database
print("💽 Connecting to ChromaDB Memory...")
try:
    chroma_client = chromadb.PersistentClient(path=os.path.join(SCRIPT_DIR, "morningstar_db"))
    collection = chroma_client.get_collection(name="daily_research")
except Exception as e:
    print(f"⚠️ Error loading database: {e}\nDid you run digest_generator.py first?")
    exit()

def query_morningstar(question):
    print(f"\n🔍 Searching your AI memory for: '{question}'...\n")
    
    # 1. Embed the user's question
    query_embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=question)['embedding']
    
    # 2. Retrieve top 3 most relevant papers from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    if not results['documents'][0]:
        print("⚠️ No documents found in your database. Let it run for a few days to build memory!")
        return
        
    # 3. Construct the Context
    context = ""
    sources = []
    for doc, meta, doc_id in zip(results['documents'][0], results['metadatas'][0], results['ids'][0]):
        context += f"Document Title: {meta['title']}\nContent: {doc}\n\n"
        sources.append(f"- {meta['title']} (Score: {meta['score']}/10) | Link: {doc_id}")
        
    # 4. The RAG Prompt
    system_prompt = f"""
    You are an expert Data Scientist and AI Assistant working for Varad at Authmind. 
    Answer the user's question based ONLY on the provided context from research papers.
    If the answer is not in the context, say "I do not have enough information in my database to answer this."
    Keep your answer concise and professional.
    
    CONTEXT:
    {context}
    """
    
    print("🧠 Synthesizing answer using Qwen2.5...\n")
    response = ollama.chat(model=LLM_MODEL, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': question}
    ])
    
    # 5. Output the result
    print("🤖 MORNINGSTAR RESPONSE:")
    print("--------------------------------------------------")
    print(response['message']['content'])
    print("--------------------------------------------------")
    print("\n📚 SOURCES USED:")
    for source in sources:
        print(source)
    print("\n")

if __name__ == "__main__":
    print("🌟 Welcome to Morningstar RAG CLI 🌟")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("Ask Morningstar: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down Morningstar. Goodbye!")
            break
        query_morningstar(user_input)