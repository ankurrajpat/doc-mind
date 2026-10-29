"""
Doc-Mind: a locally-hosted RAG app for chatting with long PDFs.

What's different from a tutorial RAG:
  1. Context-enriched chunks. Every chunk is embedded together with the
     document/section/page context it came from, so a chunk that says
     "The pressure relief valve is rated at 3200 psi." can be told apart
     from the identical sentence in a different section. On the synthetic
     60-page manual in eval_data/, this takes retrieval hit@5 from 62.5%
     to 100% (see EVALUATION.md).
  2. Optional LLM running summary. At ingest, the document is read page by
     page and a running summary is carried forward (idea borrowed from
     github.com/Rajpragur/tewtr). The summary at each point is prepended to
     that page's chunks, so early definitions are never "forgotten" by page
     40. Slower at ingest; off by default.
  3. An eval harness (eval_rag.py) so every claim above is reproducible.

Requires: Ollama running locally with `ollama run mistral`.
"""

import os
import re
import warnings
from operator import itemgetter

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
try:  # langchain-ollama (new) first, community fallback
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = None

HEADING_RE = re.compile(r"^\s*(section\s+\w+|chapter\s+\w+|appendix\s+\w+)?\s*[:\-]?\s*([A-Z][A-Z0-9 \-/]{4,60})\s*$")


def detect_section(page_text, current):
    """Best-effort section title: first ALL-CAPS-looking line on the page."""
    for line in page_text.splitlines():
        line = line.strip()
        if HEADING_RE.match(line):
            return line[:80]
    return current


def build_running_summaries(pages, llm):
    """tewtr-style: read page by page, carry one running summary forward."""
    summary = ""
    summaries = []
    for page in pages:
        prompt = (
            "You are indexing a document for search. Here is the running summary "
            "of the document so far, followed by the next page.\n\n"
            f"Summary so far:\n{summary or '(start of document)'}\n\n"
            f"Next page:\n{page[:3000]}\n\n"
            "Update the running summary in at most 4 bullet points, keeping facts "
            "(names, definitions, numbers) that later pages may refer back to:"
        )
        summary = llm.invoke(prompt)
        summaries.append(summary)
    return summaries


def process_document(pdf_file, use_context_headers, use_running_summary, progress=gr.Progress()):
    global rag_chain

    if pdf_file is None:
        yield "⚠️ Please upload a PDF first."
        return

    try:
        yield "⏳ Loading document..."
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        doc_name = os.path.basename(pdf_file.name)

        yield "✂️ Chunking text (Size: 1200, Overlap: 300)..."
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)

        llm = Ollama(model="mistral")

        page_texts = [d.page_content for d in documents]
        summaries = None
        if use_running_summary:
            yield "🧾 Building running summary page by page (slow, uses the LLM)..."
            summaries = build_running_summaries(page_texts, llm)

        chunks = []
        section = ""
        for doc in documents:
            page_num = doc.metadata.get("page", 0)
            section = detect_section(doc.page_content, section)
            for piece in text_splitter.split_documents([doc]):
                text = piece.page_content
                if use_context_headers:
                    header = f"[{doc_name} | Section: {section or 'unknown'} | Page {page_num + 1} of {len(documents)}]"
                    if summaries:
                        header += f"\n[Document so far: {summaries[page_num][:600]}]"
                    text = header + "\n" + text
                piece.page_content = text
                chunks.append(piece)

        yield "🧠 Generating embeddings (HuggingFace)..."
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        yield "🗄️ Building FAISS Vector Database..."
        vector_db = FAISS.from_documents(chunks, embeddings)

        yield "⚙️ Configuring Retrieval Engine..."
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})

        template = """You are a highly precise technical assistant. Your ONLY job is to answer the user's question based strictly on the provided Context.

        Rules:
        1. If the answer is not contained within the Context below, you must strictly say: "I cannot answer this based on the provided document."
        2. Do not use outside knowledge.
        3. ALWAYS cite the page number at the end of your answer (e.g., "[Source: Page 4]").
        4. Chunks may start with a [file | section | page] header - use it to disambiguate chunks that look similar, and to resolve references to earlier definitions.

        Previous Conversation History:
        {history}

        Context:
        {context}

        Question: {question}

        Answer:"""
        prompt = PromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(
                f"--- Chunk {i + 1} ---\n{doc.page_content}" for i, doc in enumerate(docs)
            )

        rag_chain = (
            {
                "context": itemgetter("question") | retriever | format_docs,
                "question": itemgetter("question"),
                "history": itemgetter("history"),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        yield (f"✅ Document processed: {len(documents)} pages, {len(chunks)} chunks "
               f"(context headers: {'on' if use_context_headers else 'off'}, "
               f"running summary: {'on' if use_running_summary else 'off'}). Ready to chat.")

    except Exception as e:
        yield f"❌ An error occurred: {str(e)}"


def chat_interface(user_message, history):
    global rag_chain

    if rag_chain is None:
        yield "⚠️ Please upload and process a PDF document in the Control Panel first."
        return

    formatted_history = ""
    for user_msg, bot_msg in history[-3:]:
        formatted_history += f"User: {user_msg}\nAssistant: {bot_msg}\n"

    try:
        response = ""
        for chunk in rag_chain.stream({"question": user_message, "history": formatted_history}):
            response += chunk
            yield response
    except Exception as e:
        yield f"An error occurred while generating the answer: {str(e)}"


# ==========================================
# UI
# ==========================================

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(title="Doc-Mind", theme=theme) as interface:

    gr.HTML("""
        <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%); color: white; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="margin: 0; font-size: 2.5em; font-weight: 700; letter-spacing: -1px;">📚 Doc-Mind</h1>
            <p style="margin: 8px 0 0 0; font-size: 1.1em; opacity: 0.9;">RAG for long PDFs, with measured retrieval</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=3, variant="panel"):
            gr.Markdown("### 📂 1. Knowledge Base")
            gr.Markdown("Upload a document to inject it into the AI's memory.")

            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            context_toggle = gr.Checkbox(
                label="Context-enriched chunks (recommended - see EVALUATION.md)", value=True)
            summary_toggle = gr.Checkbox(
                label="LLM running summary (slower ingest, helps on very long docs)", value=False)
            process_btn = gr.Button("🚀 Initialize Pipeline", variant="primary", size="lg")

            gr.Markdown("### 📊 2. System Logs")
            status_output = gr.Textbox(label="Status", interactive=False, lines=2)

            with gr.Accordion("⚙️ Architecture Details", open=False):
                gr.Markdown("""
                - **LLM:** Mistral (7B) via Ollama
                - **Embeddings:** `all-MiniLM-L6-v2`
                - **Vector Store:** FAISS (Local)
                - **Chunking:** 1200 chars / 300 overlap + context headers
                - **Memory:** 3-Turn Sliding Window
                - **Streaming:** True
                - **Evals:** `python eval_rag.py run` (see EVALUATION.md)
                """)

        with gr.Column(scale=7):
            chat = gr.ChatInterface(
                fn=chat_interface,
                chatbot=gr.Chatbot(height=550, show_label=False),
                textbox=gr.Textbox(placeholder="Ask a question about your document...", container=False, scale=7),
            )

    process_btn.click(fn=process_document,
                      inputs=[file_input, context_toggle, summary_toggle],
                      outputs=status_output)

if __name__ == "__main__":
    interface.launch()
