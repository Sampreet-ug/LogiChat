import chromadb
from chromadb.utils import embedding_functions
import subprocess

# Initialize ChromaDB
persist_dir = "./chroma_db"  # Ensure it is the same as in setup_chroma.py
client = chromadb.PersistentClient(path=persist_dir)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

try:
    # Attempt to retrieve the existing collection
    collection = client.get_collection(name="knowledge_base")
except ValueError:
    # If it doesn't exist, create a new collection with the embedding function
    collection = client.create_collection(name="knowledge_base", embedding_function=embed_fn)
    print("Created new knowledge_base collection.")


def search_knowledge_base(query):
    results = collection.query(query_texts=[query], n_results=1)
    #print(f"Raw Query Results: {results}")  # Debug output
    if results and results['documents'] and results['documents'][0]:
        return results['documents'][0]
    return "No relevant information found."



def ask_ollama(prompt):
    #print("Sending prompt to Ollama:\n", prompt)  # Debug print
    result = subprocess.run(["ollama", "run", "llama2"], input=prompt.encode(), capture_output=True)
    return result.stdout.decode()


# CLI Loop
if __name__ == "__main__":
    print("Welcome to the Local AI Chatbot. Type 'exit' to quit.")
    #print(client.list_collections())
    stored_docs = collection.get()
    print("Stored Data in Collection:", stored_docs)
    for collection in client.list_collections():
        print(collection.name)
    print(collection.count())  # Should return a non-zero value if data exists



    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        kb_response = search_knowledge_base(user_input)
        combined_prompt = (
            f"You are an AI assistant with access to a knowledge base. Use the following information to answer the user's question.\n\n"
            f"Knowledge Base Context:\n{kb_response}\n\n"
            f"User Question: {user_input}\n"
            f"Answer prioritizing the knowledgebase. If no answer is available in knowledge base then give your response.'"
        )
        response = ask_ollama(combined_prompt)
        print(f"Bot: {response}")
