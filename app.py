import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# -- Configuration --
DOCS_PATH = "./docs"
CHROMA_PATH = "./chroma_store"
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# -- Prompt --
custom_prompt_template = """
You are a helpful technical onboarding assistant for SAP Emarsys. Answer the user's question using the product documentation provided below.

Your answer should follow this format:

1. **High-level Summary** — Briefly explain what the feature or topic is.
2. **Step-by-Step Instructions** — If the question relates to a task or integration, explain exactly how to do it with clear numbered steps. Include command names, config settings, or interface steps if available.
3. **Details and Context** — Provide additional detail, relevant options, or variations users might encounter.
4. **References** — Mention document titles, page numbers, or section names if available.

Only use the context provided below. Do not make up any information.

---------------------
{context}
---------------------

Question: {question}

Answer:
"""


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=custom_prompt_template,
)

# -- PDF Loading & Vectorstore Creation --

@st.cache_resource
def get_or_create_vectorstore():
    if os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0:
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

    loader = GenericLoader(
        blob_loader=FileSystemBlobLoader(path=DOCS_PATH, glob="*.pdf"),
        blob_parser=PyPDFParser(),
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedding = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_PATH,
    )
    vectorstore.persist()
    return vectorstore

# -- LangChain Chain --
def get_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# -- Streamlit UI --

st.set_page_config(page_title="Product Docs RAG Chatbot", layout="centered")
st.title("📘 Product Onboarding Assistant")
st.write("Ask a question based on the product documentation in `/docs`.")

query = st.text_input("Enter your question:")
submit = st.button("Ask")

if submit and query:
    with st.spinner("Thinking..."):
        import time
        start = time.time()

        vectorstore = get_or_create_vectorstore()
        qa_chain = get_qa_chain(vectorstore)
        result = qa_chain({"query": query})

        end = time.time()

        st.markdown("### ✅ Answer")
        st.write(result["result"])

        st.markdown("### 🔗 Source References")
        for i, doc in enumerate(result["source_documents"], 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            st.write(f"{i}. **Document**: {source}, **Page**: {page}")

        st.markdown(f"⏱️ **Query took:** `{end - start:.2f}` seconds")
