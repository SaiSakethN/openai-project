import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing. Add it to your .env file or Streamlit secrets.")
    st.stop()

st.set_page_config(page_title="Transformers RAG Chatbot", page_icon="🤖")
st.title("🤖 Transformers RAG Chatbot")
st.write("Ask questions only about the Transformers document.")


@st.cache_resource
def build_vectorstore():
    url = "https://huggingface.co/docs/transformers/index"

    loader = WebBaseLoader(url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


with st.spinner("Loading and indexing the Transformers document..."):
    vectorstore = build_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are a strict question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer is not in the context, say exactly:
"I can only answer questions related to the Transformers document."

Do not use outside knowledge.
Keep the answer clear and concise.

Context:
{context}

Question:
{input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

user_question = st.text_input("What question do you have in mind?")

if user_question:
    with st.spinner("Searching the document..."):
        response = retrieval_chain.invoke({"input": user_question})
        st.write(response["answer"])
