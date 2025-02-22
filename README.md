
# LogiChat 🤖

**LogiChat** is a local AI-powered Command-Line Interface (CLI) chatbot designed to provide intelligent responses using a knowledge base of documents and log files. The chatbot runs entirely offline, ensuring privacy and security while handling sensitive data.

---

## 🚀 Features

- 🔒 **Offline Operation**: Works entirely on your local machine—no internet connection needed.
- 📚 **Multi-File Support**: Reads `.txt`, `.md`, `.pdf`, and image files (`.png`, `.jpg`, `.jpeg`).
- 🗂️ **Knowledge Base Indexing**: Efficiently stores and retrieves data using ChromaDB.
- 🧠 **AI-Powered Responses**: Provides relevant answers based on your indexed knowledge base.
- 🎯 **Easy Setup**: Quick and straightforward installation.

---

## 📂 Project Structure

```
LogiChat/
│
├── chatbot_data/                 # Knowledge base files
│   ├── sample_guide.txt
│   ├── project_info.md
│   ├── user_guide.pdf
│   └── alert_message.png
│
├── chroma_db/                    # ChromaDB storage (auto-generated)
│
├── src/                          # Source code for the chatbot
│   ├── setup_chroma.py           # Script to set up and index knowledge base
│   └── chatbot_cli.py            # Command-line chatbot interface
│
├── .gitignore                    # Specifies files Git should ignore
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── LICENSE                       # License file (MIT)
```

---

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sampreet-ug/LogiChat.git
cd LogiChat
```

### 2️⃣ Set Up the Virtual Environment

Create and activate a Python virtual environment:

```bash
# Create virtual environment
python -m venv chatbot_env

# Activate (Windows)
.\chatbot_env\Scripts\Activate

# Activate (Mac/Linux)
source chatbot_env/bin/activate
```

### 3️⃣ Install Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Knowledge Base Files

Place all your knowledge base files (`.txt`, `.md`, `.pdf`, `.png`, `.jpg`, `.jpeg`) inside the `chatbot_data/` folder.

### 5️⃣ Index Your Knowledge Base

Run the setup script to process and index your files:

```bash
python src/setup_chroma.py
```

### 6️⃣ Run the Chatbot

Once the setup is complete, launch the chatbot:

```bash
python src/chatbot_cli.py
```

---

## 💡 Usage Example

```
> How do I restart the server?
Bot:
1. Connect to the server using SSH.
2. Run: sudo systemctl restart server-name
3. Verify status with: sudo systemctl status server-name
```

---

## 🛠️ How It Works

1. **File Parsing:** Extracts text from supported file types using libraries like PyMuPDF (for PDFs) and Tesseract OCR (for images).  
2. **Embedding Creation:** Converts extracted text into embeddings using SentenceTransformers.  
3. **Storage:** Saves the embeddings in ChromaDB for efficient search and retrieval.  
4. **Query Handling:** Matches user questions with the most relevant data from the indexed knowledge base.

---

## ✅ Supported File Formats

- 📄 **Text Files:** `.txt`, `.md`
- 📚 **PDF Documents:** `.pdf`
- 🖼️ **Images (with OCR):** `.png`, `.jpg`, `.jpeg`

---

## ❗ Troubleshooting

### 🔒 Script Execution Policy Error (Windows)

If you encounter the following error:
```
.\chatbot_env\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled...
```
Run PowerShell as Administrator and execute:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### ❌ ChromaDB Collection Error

If you see this error:
```
chromadb.errors.InvalidCollectionException: Collection knowledge_base does not exist.
```
Run the indexing setup again:

```bash
python src/setup_chroma.py
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this project with proper attribution.

---

## 👥 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

---

## 📧 Contact

For questions, feedback, or collaborations, feel free to reach out at:

- GitHub: [your-username](https://github.com/sampreet-ug)

---

**Happy Chatting! 🚀**
