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
        yield "⚠️ Please upload and process a PDF using the panel on the left first."
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
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

CUSTOM_CSS = """
/* ---------- App shell ---------- */
body, .gradio-container {
    background: radial-gradient(1200px 600px at 80% -10%, rgba(124, 58, 237, 0.28), transparent 60%),
                radial-gradient(900px 500px at -10% 110%, rgba(37, 99, 235, 0.22), transparent 55%),
                #0a0e1a !important;
}
.gradio-container { max-width: 1280px !important; padding-bottom: 28px !important; }

/* ---------- Hero header ---------- */
#hero {
    display: flex; align-items: center; gap: 18px;
    padding: 26px 30px;
    background: linear-gradient(120deg, rgba(124, 58, 237, 0.16), rgba(37, 99, 235, 0.12));
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    box-shadow: 0 12px 40px rgba(2, 6, 23, 0.55), inset 0 1px 0 rgba(255,255,255,0.06);
    backdrop-filter: blur(6px);
    margin-bottom: 20px;
}
#hero .logo {
    width: 58px; height: 58px; flex: 0 0 58px;
    display: flex; align-items: center; justify-content: center;
    font-size: 30px;
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.45);
}
#hero h1 {
    margin: 0; font-size: 2.1em; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #e9d5ff, #93c5fd);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
#hero p { margin: 4px 0 0 0; color: #94a3b8; font-size: 0.98em; }
#hero .badges { margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
#hero .badge {
    padding: 5px 12px; font-size: 0.72em; font-weight: 600; letter-spacing: 0.4px;
    color: #c4b5fd; background: rgba(124, 58, 237, 0.14);
    border: 1px solid rgba(167, 139, 250, 0.35); border-radius: 999px;
    white-space: nowrap;
}

/* ---------- Panels & sections ---------- */
#control-panel {
    background: rgba(15, 23, 42, 0.72) !important;
    border: 1px solid rgba(148, 163, 184, 0.16) !important;
    border-radius: 18px !important;
    box-shadow: 0 12px 32px rgba(2, 6, 23, 0.45);
}
.step-label {
    display: flex; align-items: center; gap: 10px;
    margin: 18px 4px 6px 4px; color: #e2e8f0 !important;
    font-weight: 700; font-size: 0.95em;
}
.step-num {
    width: 24px; height: 24px; flex: 0 0 24px;
    display: inline-flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: #fff; border-radius: 8px; font-size: 0.78em; font-weight: 700;
}
.step-hint { margin: 0 4px 10px 38px; color: #64748b !important; font-size: 0.82em; }

/* ---------- Chat column ---------- */
#chat-col {
    background: rgba(15, 23, 42, 0.72) !important;
    border: 1px solid rgba(148, 163, 184, 0.16) !important;
    border-radius: 18px !important;
    padding: 18px !important;
    box-shadow: 0 12px 32px rgba(2, 6, 23, 0.45);
}
.chat-title {
    margin: 2px 4px 12px 4px; color: #e2e8f0 !important;
    font-weight: 700; font-size: 1.05em;
}

/* Chat bubbles */
.message.user, .message-row.user-row .message {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: #fff !important; border-radius: 16px 16px 4px 16px !important;
    border: none !important;
}
.message.bot, .message-row.bot-row .message {
    background: rgba(30, 41, 59, 0.9) !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: 16px 16px 16px 4px !important;
}

/* ---------- Primary button ---------- */
#process-btn button, button.primary {
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%) !important;
    border: none !important; color: #fff !important; font-weight: 700 !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4);
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
#process-btn button:hover, button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 26px rgba(124, 58, 237, 0.55);
}

/* ---------- Field labels (plain, no pill background) ---------- */
label.float, span[class*="block-label"], .gradio-container .block-label {
    background: transparent !important;
    color: #94a3b8 !important;
    border: none !important; box-shadow: none !important;
    font-weight: 600 !important; text-transform: none !important;
}

/* ---------- Misc ---------- */
#footer-note { text-align: center; color: #475569; font-size: 0.78em; margin-top: 18px; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.3); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }
"""

# Force dark mode so the custom shell always renders as designed.
dark_mode_js = """
function forceDark() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

# Gradio 6 moved theme/css/js from Blocks() to launch(); keep 4/5 compatibility.
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_blocks_kwargs = {} if _GRADIO_MAJOR >= 6 else {"theme": theme, "css": CUSTOM_CSS, "js": dark_mode_js}
_launch_kwargs = {"theme": theme, "css": CUSTOM_CSS, "js": dark_mode_js} if _GRADIO_MAJOR >= 6 else {}

with gr.Blocks(title="Doc-Mind", **_blocks_kwargs) as interface:

    gr.HTML("""
        <div id="hero">
            <div class="logo">📚</div>
            <div>
                <h1>Doc-Mind</h1>
                <p>Chat with long PDFs - RAG with measured retrieval, running locally on your machine</p>
            </div>
            <div class="badges">
                <span class="badge">MISTRAL 7B</span>
                <span class="badge">FAISS</span>
                <span class="badge">100% LOCAL</span>
            </div>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=3, elem_id="control-panel"):
            gr.HTML('<div class="step-label"><span class="step-num">1</span>Upload document</div>'
                    '<p class="step-hint">Drop a PDF to load it into the knowledge base.</p>')

            file_input = gr.File(label="PDF file", file_types=[".pdf"])

            gr.HTML('<div class="step-label"><span class="step-num">2</span>Retrieval options</div>'
                    '<p class="step-hint">How chunks get context before embedding.</p>')
            context_toggle = gr.Checkbox(
                label="Context-enriched chunks (recommended - see EVALUATION.md)", value=True)
            summary_toggle = gr.Checkbox(
                label="LLM running summary (slower ingest, helps on very long docs)", value=False)

            gr.HTML('<div class="step-label"><span class="step-num">3</span>Initialize</div>')
            process_btn = gr.Button("🚀  Build Knowledge Base", variant="primary",
                                    size="lg", elem_id="process-btn")

            gr.HTML('<div class="step-label">📡 System status</div>')
            status_output = gr.Textbox(show_label=False, interactive=False, lines=2,
                                       placeholder="Pipeline status will appear here...")

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

        with gr.Column(scale=7, elem_id="chat-col"):
            gr.HTML('<div class="chat-title">💬 Chat with your document</div>')
            chat = gr.ChatInterface(
                fn=chat_interface,
                chatbot=gr.Chatbot(height=560, show_label=False),
                textbox=gr.Textbox(placeholder="Ask a question about your document...",
                                   container=False, scale=7),
            )

    gr.HTML('<div id="footer-note">Doc-Mind runs entirely on localhost - your documents never leave this machine.</div>')

    process_btn.click(fn=process_document,
                      inputs=[file_input, context_toggle, summary_toggle],
                      outputs=status_output)

if __name__ == "__main__":
    interface.launch(**_launch_kwargs)
