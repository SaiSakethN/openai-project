import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

print("OPENAI:", os.getenv("OPENAI_API_KEY") is not None)
print("LANGSMITH:", os.getenv("LANGSMITH_API_KEY") is not None)
print("TRACING:", os.getenv("LANGSMITH_TRACING"))
print("PROJECT:", os.getenv("LANGSMITH_PROJECT"))
print("ENDPOINT:", os.getenv("LANGSMITH_ENDPOINT"))

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","your are a help assistant, please respoond to questions asked"),
        ("user","Question:{question}")
    ]
)

st.title("Langchain With Gemma")
input_text=st.text_input("What question is have in mind")

llm=Ollama(model="gemma:2b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke(({"question":input_text})))