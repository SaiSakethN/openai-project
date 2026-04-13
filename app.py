import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing. Add it to your .env file or Streamlit secrets.")
    st.stop()

st.set_page_config(page_title="Dynamic Topic RAG Chatbot", page_icon="🤖")
st.title("🤖 Dynamic Topic RAG Chatbot")
st.write("Enter any topic, load its knowledge from Wikipedia, and then ask questions about it.")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "current_topic" not in st.session_state:
    st.session_state.current_topic = None

if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False


def build_vectorstore_from_topic(topic: str):
    loader = WikipediaLoader(query=topic, load_max_docs=3)
    docs = loader.load()

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


topic = st.text_input("Enter a topic", placeholder="Example: Retrieval Augmented Generation")

if st.button("Load Topic Knowledge"):
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner(f"Loading knowledge for '{topic}'..."):
            vectorstore = build_vectorstore_from_topic(topic)

            if vectorstore is None:
                st.error("No documents were found for this topic. Try another topic.")
            else:
                st.session_state.vectorstore = vectorstore
                st.session_state.current_topic = topic
                st.session_state.docs_loaded = True
                st.success(f"Knowledge loaded for: {topic}")


if st.session_state.docs_loaded and st.session_state.vectorstore is not None:
    st.info(f"Current topic: {st.session_state.current_topic}")

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are a strict question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer is not in the context, say exactly:
"I can only answer questions related to the loaded topic."

Keep the answer clear, accurate, and easy to understand.
Do not use outside knowledge.

Context:
{context}

Question:
{input}
""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    user_question = st.text_input(
        "Ask a question about the loaded topic",
        placeholder="Example: What are the main applications of RAG?"
    )

    if user_question:
        with st.spinner("Searching and generating answer..."):
            response = retrieval_chain.invoke({"input": user_question})
            st.write(response["answer"])
else:
    st.caption("Load a topic first to start asking questions.")
