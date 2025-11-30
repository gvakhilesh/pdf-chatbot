import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os
import time
import json
from datetime import datetime
import shutil
import glob
from dotenv import load_dotenv

load_dotenv()

# ======================================================
# 1. CHAT HISTORY PERSISTENCE
# ======================================================
HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)


# ======================================================
# 2. STREAMLIT CONFIG + UI STYLING
# ======================================================
st.set_page_config(page_title="PDF Chatbot (RAG + Gemini)", layout="wide")

dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)

if dark_mode:
    st.markdown("""
    <style>
        body { background-color: #000000 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)


st.markdown("""
<style>
.chat-row { display: flex; align-items: flex-start; margin-bottom: 14px; }
.chat-avatar { width: 42px; height: 42px; border-radius: 50%; margin-right: 12px; }
.chat-bubble { padding: 12px 16px; border-radius: 14px; max-width: 78%; font-size: 16px; }
.user-bubble { background-color: #DCF8C6; color: black; }
.ai-bubble { background-color: #EDEDED; color: black; }
.timestamp { font-size: 12px; opacity: 0.6; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ======================================================
# 3. RAG UTILS
# ======================================================
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def build_chain(chat_history, retriever, prompt, model):
    history_text = "\n".join([f"{role}: {msg}" for role, msg, _ in chat_history])

    chain_inputs = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
        "history": lambda _: history_text
    })

    parser = StrOutputParser()
    return chain_inputs | prompt | model | parser


# ======================================================
# 4. SIDEBAR — PDF Upload
# ======================================================
st.sidebar.header("📄 PDF Options")

uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# Cleanup button
if st.sidebar.button("🧹 Clean All DB Folders"):
    for folder in glob.glob("chroma_db_*"):
        try:
            shutil.rmtree(folder)
        except:
            pass
    st.sidebar.success("Old DB folders removed!")


# ======================================================
# 5. CREATE DB ONLY WHEN NEW PDF IS UPLOADED
# ======================================================
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

# When new PDF uploaded
if uploaded_pdf and not st.session_state.pdf_loaded:

    # Save new PDF
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    # Create unique DB only ONCE
    persist_dir = f"./chroma_db_{int(time.time())}"
    st.session_state.persist_dir = persist_dir

    loader = PyPDFLoader("uploaded.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    db = Chroma.from_documents(
        docs,
        embeddings,
        collection_name="pdf_docs",
        persist_directory=persist_dir
    )

    st.session_state.retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    st.session_state.pdf_loaded = True

    st.sidebar.success("PDF processed and Vector DB created!")


# Stop if no PDF
if st.session_state.retriever is None:
    st.info("📥 Upload a PDF to start chatting.")
    st.stop()


# ======================================================
# 6. LLM + PROMPT
# ======================================================
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

prompt = PromptTemplate(
    template="""
You answer ONLY based on the provided PDF context.

Chat History:
{history}

Context:
{context}

Question:
{question}

If the answer is NOT found, reply:
"I don't know based on the provided document."
""",
    input_variables=["context", "question", "history"]
)


# ======================================================
# 7. SESSION HISTORY
# ======================================================
if "history" not in st.session_state:
    st.session_state.history = load_history()


# ======================================================
# 8. MAIN CHAT UI
# ======================================================
st.title("🤖 PDF Chatbot — RAG + Gemini")

for role, msg, timestamp in st.session_state.history:
    avatar = (
        "https://i.imgur.com/5cQ5YtE.png" if role == "User"
        else "https://i.imgur.com/1Xq9BiH.png"
    )
    bubble_class = "user-bubble" if role == "User" else "ai-bubble"

    st.markdown(
        f"""
        <div class="chat-row">
            <img class="chat-avatar" src="{avatar}">
            <div>
                <div class="chat-bubble {bubble_class}">{msg}</div>
                <div class="timestamp">{timestamp}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================================================
# 9. CHAT INPUT
# ======================================================
user_input = st.chat_input("Ask something about your PDF...")

if user_input:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append(("User", user_input, timestamp))
    save_history(st.session_state.history)

    with st.spinner("🧠 Thinking..."):
        rag_chain = build_chain(
            st.session_state.history,
            st.session_state.retriever,  # <--- uses SAME DB every time
            prompt,
            model,
        )
        result = rag_chain.invoke(user_input)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append(("AI", result, timestamp))
    save_history(st.session_state.history)

    st.rerun()
