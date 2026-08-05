📚 DocMind : Enterprise-Grade RAG Pipeline

An advanced, locally-hosted Retrieval-Augmented Generation (RAG) application that allows users to chat with complex PDF documents. This project was built to demonstrate modern AI orchestration, strict context grounding, and latency optimization using LangChain's Expression Language (LCEL).

🚀 Features

Strict Context Grounding: The LLM is mathematically restricted from hallucinating. If the answer is not in the uploaded document, it explicitly refuses to answer.

Real-Time Streaming: Implemented asynchronous token streaming to drastically reduce Time-To-First-Token (TTFT).

Conversational Memory: Utilizes a sliding-window memory architecture so the agent remembers previous conversational turns for follow-up questions.

100% Local & Private: Powered by Mistral-7B via Ollama and all-MiniLM-L6-v2 embeddings, ensuring no data ever leaves the local machine.

🧠 System Architecture

Ingestion: Documents are parsed using PyPDFLoader and split using a RecursiveCharacterTextSplitter (1200 chunk size / 300 overlap) to preserve semantic sentence boundaries.

Vector Store: Chunks are vectorized using HuggingFace embeddings and stored in a highly efficient, in-memory FAISS vector database.

Retrieval: The system performs a cosine-similarity search (k=5) against the user's query.

Orchestration (LCEL): The retrieved context, conversation history, and user query are dynamically routed into a strict Prompt Template using itemgetter, then piped into the LLM and out through a StrOutputParser.

⚙️ Installation & Setup

Clone the repository:

git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME


Create a virtual environment:

python -m venv venv
venv\Scripts\activate  # On Windows


Install dependencies:

pip install langchain langchain-community langchain-huggingface faiss-cpu pypdf gradio sentence-transformers


Install Ollama & Download Model:
Download Ollama and run:

ollama run mistral


Launch the application:

python rag_app.py
