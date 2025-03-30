import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions
import subprocess
import re

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
    """Retrieves only relevant past chat history."""
    results = chat_history.query(query_texts=[user_input], n_results=num_results)

    past_chats = results.get("documents", [[]])[0]  
    metadatas = results.get("metadatas", [[]])[0]  

    # Ensure only relevant history is included
    relevant_chats = []
    for i in range(len(past_chats)):
        if isinstance(metadatas[i], dict) and "response" in metadatas[i]:
            relevant_chats.append(f"User: {str(past_chats[i])}\nBot: {metadatas[i]['response']}")

    return "\n".join(relevant_chats) if relevant_chats else "No relevant chat history found."


def search_knowledge_base(query, threshold=0.7):
    """Searches the knowledge base and returns relevant info only if similarity is high."""
    results = knowledge_base.query(query_texts=[query], n_results=1)

    if results and results["documents"] and len(results["documents"][0]) > 0:
        kb_match = results["documents"][0][0]  # Get the first result
        similarity_score = results["distances"][0][0] if "distances" in results else None

        # Only return KB response if similarity score is high
        if isinstance(kb_match, str) and similarity_score is not None and similarity_score <= threshold:
            return kb_match  # Return relevant knowledge base entry

    return None  # Return None if no relevant match

def ask_ollama(prompt):
    """Sends a query to Ollama's model."""
    result = subprocess.run(["ollama", "run", "llama2"], input=prompt.encode(), capture_output=True)
    return result.stdout.decode()

def is_url(text):
    url_pattern = re.compile(r"https?://\S+")
    return bool(url_pattern.match(text))

def fetch_clean_webpage(url):
    """Fetches webpage content and extracts meaningful text."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unnecessary tags
        for tag in soup(["script", "style", "footer", "nav"]):
            tag.extract()

        # Extract important sections
        headers = " ".join([h.get_text() for h in soup.find_all(["h1", "h2"])])
        paragraphs = " ".join([p.get_text() for p in soup.find_all("p")])

        content = headers + "\n" + paragraphs
        content = re.sub(r'\s+', ' ', content)  # Remove excessive whitespace
        return content.strip()
    
    except Exception as e:
        return f"Error fetching content: {e}"

def summarize_text(text, num_sentences=5):
    """Uses TextRank (LSA) to extract key points."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)


# CLI Loop
if __name__ == "__main__":
    print("Welcome to the Local AI Chatbot. Type 'exit' to quit.")
    
    chat_id = "user123"  # This can be session-based or user-specific and could be set based on the authentication(to be implemented).

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        if is_url(user_input):
            print("Fetching content from the URL...")
            webpage_text = fetch_clean_webpage(user_input)

            if len(webpage_text) > 5000:  # Summarize only if too long
                webpage_text = summarize_text(webpage_text, num_sentences=5)

            combined_prompt = f"Summarize the following content:\n\n{webpage_text}\n\n"
            response = ask_ollama(combined_prompt)
        else:
            kb_response = search_knowledge_base(user_input)

            if kb_response:  
                # If KB has a relevant answer, prioritize it
                combined_prompt = (
                    f"You are an AI assistant. Answer using only the following knowledge base information:\n\n"
                    f"{kb_response}\n\n"
                    f"If this information does not answer the question, say 'I don't know'."
                )
            else:
                past_chats = retrieve_chat_memory(user_input, chat_id)
                
                if past_chats and "No relevant chat history found." not in past_chats:
                    # Use past conversations only if they are relevant
                    combined_prompt = (
                        f"You are an AI assistant. Here is past chat history:\n\n"
                        f"{past_chats}\n\n"
                        f"Use this to guide your response, but only if relevant."
                    )
                else:
                    # If no relevant KB or past chat, use general LLM
                    combined_prompt = f"You are an AI assistant. Answer the following question:\n\nUser: {user_input}\nBot:"

            response = ask_ollama(combined_prompt)

        print(f"Bot: {response}")

        store_chat_memory(user_input, response, chat_id)