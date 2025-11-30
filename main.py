from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
import os
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



result = model.invoke('What is the capital of India')

print(result.content)