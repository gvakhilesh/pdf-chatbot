from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# -----------------------------------------
# Helper to format retrieved documents
# -----------------------------------------
def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


# -----------------------------------------
# Load LLM
# -----------------------------------------
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# -----------------------------------------
# Prompt Template
# -----------------------------------------
prompt = PromptTemplate(
    template="""
You are a helpful assistant that answers questions using ONLY the provided PDF context.

Chat History:
{history}

Context:
{context}

Question:
{question}

If the context does NOT contain the answer, respond with:
"I don't know based on the provided document."
""",
    input_variables=["context", "question", "history"]
)

# -----------------------------------------
# Load PDF and Split Documents
# -----------------------------------------
file_path = "scripttt.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

# -----------------------------------------
# Embeddings + Chroma DB
# -----------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

db = Chroma.from_documents(
    docs,
    embeddings,
    collection_name="pdf_docs",
    persist_directory="./chroma_db"
)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# -----------------------------------------
# Build RAG Chain
# -----------------------------------------
def build_chain(chat_history):
    history_text = "\n".join([f"{role}: {msg}" for role, msg in chat_history])

    chain_inputs = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
        "history": lambda _: history_text
    })

    parser = StrOutputParser()

    return chain_inputs | prompt | model | parser


# -----------------------------------------
# Chat Loop
# -----------------------------------------
print("PDF Chatbot Ready! Type 'exit' to quit.\n")

chat_history = []

while True:
    user_input = input("You: ")

    if user_input.lower().strip() == "exit":
        break

    rag_chain = build_chain(chat_history)

    result = rag_chain.invoke(user_input)

    print("\n=== MODEL RESPONSE ===")
    print(result)
    print("======================\n")

    chat_history.append(("User", user_input))
    chat_history.append(("AI", result))
