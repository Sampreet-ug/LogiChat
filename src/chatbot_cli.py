import chromadb
from chromadb.utils import embedding_functions
import subprocess

# Initialize ChromaDB
client = chromadb.Client()
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

try:
    collection = client.get_collection(name="knowledge_base", embedding_function=embed_fn)
except chromadb.errors.InvalidCollectionException:
    collection = client.create_collection(name="knowledge_base", embedding_function=embed_fn)
    print("Created new knowledge_base collection.")

def search_knowledge_base(query):
    results = collection.query(query_texts=[query], n_results=1)
    if results and results['documents']:
        return results['documents'][0]
    return "No relevant information found."

def ask_ollama(prompt):
    result = subprocess.run(["ollama", "run", "llama2"], input=prompt.encode(), capture_output=True)
    return result.stdout.decode()

# CLI Loop
if __name__ == "__main__":
    print("Welcome to the Local AI Chatbot. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        kb_response = search_knowledge_base(user_input)
        combined_prompt = f"Context: {kb_response}\n\nUser: {user_input}\nResponse:"
        response = ask_ollama(combined_prompt)
        print(f"Bot: {response}")
