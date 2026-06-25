# 📄 PDF Chatbot using LangChain & RAG

A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about PDF documents using LangChain, FAISS, HuggingFace Embeddings, Groq LLM, and Streamlit.

## 🚀 Live Demo

Add your Streamlit URL here:

https://pdf-rag-chatbot-3tmwuq88awvxalklnu8o4x.streamlit.app/

## ✨ Features

* Upload and process PDF documents
* Text chunking using LangChain
* HuggingFace Embeddings
* FAISS Vector Database
* Context-aware question answering
* Groq LLM integration
* Interactive Streamlit UI

## 🛠️ Tech Stack

* Python
* LangChain
* FAISS
* HuggingFace Embeddings
* Groq
* Streamlit

## 📂 Project Structure

```text
PDF_CHATBOT/
│
├── pdf/
├── faiss_index/
├── create_vector_db.py
├── streamlit_app.py
├── README.md
├── pyproject.toml
└── uv.lock
```

## ⚙️ Installation

```bash
git clone https://github.com/vishal8248/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` file:

```env
GROQ_API_KEY=your_api_key
```

Run:

```bash
streamlit run streamlit_app.py
```

## 📸 Screenshot

Add a screenshot of your chatbot here.

## 👨‍💻 Author

Vishal Mali
