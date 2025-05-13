# import asyncio
# import nest_asyncio
# nest_asyncio.apply()

from dotenv import load_dotenv
import os
import streamlit as st
import pickle
import time
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.9, max_tokens = 500)
# file_path = "vector_store_huggingFace.pkl"
# index_folder = "faiss_index"

st.title("Equity Research Tool 📈")

st.sidebar.title("Stocks Articles URLs")

raw_urls = [st.sidebar.text_input(f"Enter URL {i+1} of the article to analyse") for i in range(3)]
urls = [u.strip() for u in raw_urls if u.strip()]

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

file_path = "vector_store_huggingFace.pkl"
# vector_store_huggingFace = None

if process_url_clicked:
  # load data
  loader = UnstructuredURLLoader(urls = urls)
  main_placeholder.text("Data Loading.... Started....✅✅✅✅ ")
  data  = loader.load()
  
  # split data
  text_splitter = RecursiveCharacterTextSplitter(
      separators = [ '\n\n' , '\n' , '.' , ','],
      chunk_size = 1000,
      chunk_overlap = 200,
  )
  main_placeholder.text("Text Splitter.... Started....✅✅✅✅ ")

  docs = text_splitter.split_documents(data)

  # create embeddings 
  embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
    )
  # store in faiss index 
  vector_store = FAISS.from_documents(docs,embeddings)
#   vector_store_huggingFace.save_local(index_folder)
  main_placeholder.text("✅ FAISS index saved. Ready for questions!")
  
  main_placeholder.text("Embedding Vector Started Building....✅✅✅✅ ")
  # time.sleep(2)

  with open(file_path, "wb") as f:
    pickle.dump(vector_store , f)

query = main_placeholder.text_input("Question:")
if query:
    # 1) make sure the vector‐store exists
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_store = pickle.load(f)

        # 2) build & run the chain
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
        )
        result = chain.invoke({"question": query}, return_only_outputs=True)

        # 3) display
        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "").strip()
        if sources:
            st.subheader("SOURCES:")
            for src in sources.split("\n"):
                st.write(src)
    else:
        st.error("No vector‑store found. Please click “Process URLs” first.")
