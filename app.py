import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to questions asked."),
        ("user", "Question: {question}")
    ]
)

st.title("LangChain With OpenAI")
input_text = st.text_input("What question do you have in mind?")

llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
