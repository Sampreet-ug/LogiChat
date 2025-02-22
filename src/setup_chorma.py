import chromadb
from chromadb.utils import embedding_functions
import os
import fitz  # For PDF extraction
from PIL import Image
import pytesseract

# Initialize client and embedding function
client = chromadb.Client()
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# PDF extraction function
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Image text extraction function
def extract_text_from_image(file_path):
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
    collection = client.create_collection(name="knowledge_base", embedding_function=embed_fn)
    docs = load_documents("chatbot_data")
    for idx, doc in enumerate(docs):
        collection.add(
            documents=[doc],
            ids=[str(idx)]
        )
    print("Knowledge base indexed successfully.")
    return collection

# Main execution
if __name__ == "__main__":
    collection = store_embeddings()
    print("Available collections:")
    print(client.list_collections())
