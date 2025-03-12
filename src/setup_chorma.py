import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import os
import fitz
from PIL import Image
import pytesseract

# Initialize client and embedding function
client = chromadb.PersistentClient(path="chroma_db/")

embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Ensure collection exists
try:
    collection = client.get_collection(name="knowledge_base")
except Exception as e:
    print(f"Collection not found: {e}")
    collection = client.create_collection(name="knowledge_base", embedding_function=embed_fn)
    print("Created new knowledge_base collection.")

# PDF extraction function
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Image text extraction function
def extract_text_from_image(file_path):
    print(pytesseract.image_to_string(Image.open(file_path)))
    return pytesseract.image_to_string(Image.open(file_path))

# Load documents from different file types
def load_documents(folder):
    documents = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filename.endswith(('.txt', '.md')):
            with open(filepath, 'r', encoding='utf-8') as file:
                documents.append(file.read())
        elif filename.endswith('.pdf'):
            documents.append(extract_text_from_pdf(filepath))
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            documents.append(extract_text_from_image(filepath))
    return documents


# Store embeddings in collection
def store_embeddings():
    # Recreate collection with embedding function applied
    

    docs = load_documents("chatbot_data")
    for idx, doc in enumerate(docs):
        collection.add(
            documents=[doc],
            ids=[str(idx)]
        )
    print("Knowledge base indexed successfully.")
    all_docs = collection.get()
    print(f"Documents in collection: {all_docs['documents']}")
    return collection

def store_documents_with_manual_embeddings():
    documents = [
        "IBM Sterling OMS is an order management system designed for enterprise-level solutions.",
        "ChromaDB is a vector database optimized for AI and embedding-based searches.",
        "Llama2 is a powerful open-source large language model developed for text generation."
    ]

    # Generate embeddings manually
    embeddings = embed_fn(documents)
    print("Generated Embeddings:", embeddings)  # Debug: Ensure embeddings are created

    if embeddings is None or len(embeddings) == 0:
        print("Embeddings not generated properly. Check embedding function.")
        return

    # Add documents with explicit embeddings
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(documents))]
    )
    print("Documents successfully added with manual embeddings.")

# Main execution
if __name__ == "__main__":
    #store_documents_with_manual_embeddings()
    store_embeddings()
    # Check if documents exist in the collection
    print("Available collections:")
    for collection in client.list_collections():
        print(collection.name)
    
    print(collection.count())  # Should return a non-zero value if data exists

    
    



