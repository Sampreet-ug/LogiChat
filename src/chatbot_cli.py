import chromadb
from chromadb.utils import embedding_functions
import subprocess

# Initialize ChromaDB
persist_dir = "./chroma_db"
client = chromadb.PersistentClient(path=persist_dir)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Ensure knowledge base collection exists
try:
    knowledge_base = client.get_collection(name="knowledge_base")
except ValueError:
    knowledge_base = client.create_collection(name="knowledge_base", embedding_function=embed_fn)
    print("Created new knowledge_base collection.")

# Ensure chat history collection exists
try:
    chat_history = client.get_collection(name="chat_history")
except ValueError:
    chat_history = client.create_collection(name="chat_history", embedding_function=embed_fn)
    print("Created new chat_history collection.")


def store_chat_memory(user_input, bot_response, chat_id):
    """Stores past chat history in ChromaDB."""
    chat_history.add(
        documents=[user_input],
        metadatas=[{"response": bot_response, "chat_id": chat_id}],
        ids=[f"{chat_id}-{len(chat_history.get()['documents'])}"]
    )


def retrieve_chat_memory(user_input, chat_id, num_results=3):
    """Retrieves past relevant chat history for context."""
    results = chat_history.query(query_texts=[user_input], n_results=num_results)

    past_chats = results.get("documents", [[]])[0]  # Ensure it's a list and take the first element
    metadatas = results.get("metadatas", [[]])[0]  # Ensure it's a list and take the first element

    # Debugging output
    #print(f"DEBUG: past_chats = {past_chats}")
    #print(f"DEBUG: metadatas = {metadatas}")

    if not past_chats or not metadatas:
        return "No relevant chat history found."

    # Ensure metadatas[i] is a dictionary and handle cases where it’s missing a response
    return "\n".join([
        f"User: {str(past_chats[i])}\nBot: {metadatas[i].get('response', 'No response recorded')}" 
        if isinstance(metadatas[i], dict) else f"User: {str(past_chats[i])}\nBot: No response recorded"
        for i in range(len(past_chats))
    ])


def search_knowledge_base(query):
    """Searches the knowledge base for relevant information."""
    results = knowledge_base.query(query_texts=[query], n_results=1)
    if results and results["documents"] and results["documents"][0]:
        return results["documents"][0]
    return "No relevant information found."


def ask_ollama(prompt):
    """Sends a query to Ollama's model."""
    result = subprocess.run(["ollama", "run", "llama2"], input=prompt.encode(), capture_output=True)
    return result.stdout.decode()


# CLI Loop
if __name__ == "__main__":
    print("Welcome to the Local AI Chatbot. Type 'exit' to quit.")
    
    stored_kb_docs = knowledge_base.get()
    print("Stored Data in Knowledge Base:", stored_kb_docs)

    for collection in client.list_collections():
        print(f"Collection: {collection.name}")
    
    chat_id = "user123"  # This can be session-based or user-specific and could be set based on the authentication(to be implemented).

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        kb_response = search_knowledge_base(user_input)
        past_chats = retrieve_chat_memory(user_input, chat_id)

        combined_prompt = (
            f"You are an AI assistant with access to a knowledge base and chat history.\n\n"
            f"Past Conversations:\n{past_chats}\n\n"
            f"Knowledge Base Context:\n{kb_response}\n\n"
            f"User Question: {user_input}\n"
            f"Answer prioritizing the knowledge base. If no answer is available, use chat history to guide the response."
        )

        response = ask_ollama(combined_prompt)
        print(f"Bot: {response}")

        store_chat_memory(user_input, response, chat_id)
