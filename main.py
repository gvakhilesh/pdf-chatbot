from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
import os
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

prompt = PromptTemplate(
    template="""
      You are a helpful assistant to chat with documents.
      Answer ONLY from the provided context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

file_path = "C:\\Users\\akhil\\Downloads\\pdf_chatbot\\NISM XXI A PMS Short Notes.PDF"
loader = PyMuPDF4LLMLoader(file_path)

result = model.invoke('What is the capital of India')

print(result.content)