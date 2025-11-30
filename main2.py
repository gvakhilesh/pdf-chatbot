import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

load_dotenv()


# OpenAI-compatible HuggingFace endpoint
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
    temperature=0.2,
)

prompt = PromptTemplate(
    template="""
    You are a helpful PDF assistant.
    Answer ONLY using the given context.
    If you don't know, say "I don't know".

    Context:
    {context}

    Question: {question}
    """,
    input_variables=["context", "question"]
)

# Load PDF
file_path = r"NSDLe-CAS_103802213_SEP_2025.PDF"
loader = PyPDFLoader(file_path)
docs = loader.load()

context = "\n\n".join([d.page_content for d in docs[:3]])  # sample

message = prompt.format(context=context, question="Explain the document summary.")

response = llm.invoke(message)

print("\n=== MODEL RESPONSE ===\n")
print(response.content)
