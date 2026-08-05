import os
import warnings
from operator import itemgetter

# Suppress warnings for a clean terminal
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Global variable to store the retrieval chain
rag_chain = None

def process_document(pdf_file):
    global rag_chain
    
    if pdf_file is None:
        return "⚠️ Please upload a PDF first."
        
    try:
        yield "⏳ Loading document..." 
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        
        yield "✂️ Chunking text (Size: 1200, Overlap: 300)..."
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
        chunks = text_splitter.split_documents(documents)
        
        yield "🧠 Generating embeddings (HuggingFace)..."
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        yield "🗄️ Building FAISS Vector Database..."
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        yield "⚙️ Configuring Retrieval Engine..."
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        
        llm = Ollama(model="mistral")
        
        template = """You are a highly precise technical assistant. Your ONLY job is to answer the user's question based strictly on the provided Context. 
        
        Rules:
        1. If the answer is not contained within the Context below, you must strictly say: "I cannot answer this based on the provided document."
        2. Do not use outside knowledge. 
        3. ALWAYS cite the page number at the end of your answer (e.g., "[Source: Page 4]").

        Previous Conversation History:
        {history}

        Context: 
        {context}
        
        Question: {question}
        
        Answer:"""
        prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):
            formatted_chunks = []
            for i, doc in enumerate(docs):
                page_num = doc.metadata.get('page', 0) + 1
                formatted_chunks.append(f"--- Chunk {i+1} (Page {page_num}) ---\n{doc.page_content}")
            return "\n\n".join(formatted_chunks)
            
        rag_chain = (
            {
                "context": itemgetter("question") | retriever | format_docs, 
                "question": itemgetter("question"),
                "history": itemgetter("history")
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        yield "✅ Document processed successfully! The AI is ready to chat."
    
    except Exception as e:
        yield f"❌ An error occurred: {str(e)}"

def chat_interface(user_message, history):
    global rag_chain
    
    if rag_chain is None:
        yield "⚠️ Please upload and process a PDF document in the Control Panel first."
        return
    
    # Extract the last 3 messages from Gradio's history to prevent context overflow
    formatted_history = ""
    for user_msg, bot_msg in history[-3:]: 
        formatted_history += f"User: {user_msg}\nAssistant: {bot_msg}\n"
        
    try:
        # Yield the response chunk-by-chunk to create a typewriter streaming effect
        response = ""
        for chunk in rag_chain.stream({"question": user_message, "history": formatted_history}):
            response += chunk
            yield response
            
    except Exception as e:
        yield f"An error occurred while generating the answer: {str(e)}"

# ==========================================
# 🎨 PREMIUM GRADIO UI DESIGN
# ==========================================

# Use a professional theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(title="Doc-AI Researcher", theme=theme) as interface:
    
    # Custom HTML Banner
    gr.HTML("""
        <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%); color: white; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="margin: 0; font-size: 2.5em; font-weight: 700; letter-spacing: -1px;">📚 Doc-AI Researcher</h1>
            <p style="margin: 8px 0 0 0; font-size: 1.1em; opacity: 0.9;">Enterprise-Grade Agentic RAG Pipeline</p>
        </div>
    """)
    
    with gr.Row():
        # LEFT CONTROL PANEL (25% width)
        with gr.Column(scale=3, variant="panel"):
            gr.Markdown("### 📂 1. Knowledge Base")
            gr.Markdown("Upload a document to inject it into the AI's memory.")
            
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("🚀 Initialize Pipeline", variant="primary", size="lg")
            
            gr.Markdown("### 📊 2. System Logs")
            # FIX: Removed the unsupported show_copy_button argument
            status_output = gr.Textbox(label="Status", interactive=False, lines=2)
            
            # Collapsible tech stack section for recruiters
            with gr.Accordion("⚙️ Architecture Details", open=False):
                gr.Markdown("""
                - **LLM:** Mistral (7B) via Ollama
                - **Embeddings:** `all-MiniLM-L6-v2`
                - **Vector Store:** FAISS (Local)
                - **Memory:** 3-Turn Sliding Window
                - **Streaming:** True
                """)
                
        # RIGHT CHAT AREA (75% width)
        with gr.Column(scale=7):
            chat = gr.ChatInterface(
                fn=chat_interface,
                chatbot=gr.Chatbot(
                    height=550, 
                    show_label=False,
                    avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/4712/4712027.png") # Robot icon for AI
                ),
                textbox=gr.Textbox(placeholder="Ask a question about your document...", container=False, scale=7),
            )
            
    # Button Click Event
    process_btn.click(fn=process_document, inputs=file_input, outputs=status_output)

if __name__ == "__main__":
    interface.launch()