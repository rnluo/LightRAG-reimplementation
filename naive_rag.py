# Implementation of naive RAG architecture as a comparison

from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import TokenTextSplitter

from prompt import *
import rag

def naive_rag(
        input: str,
        context:str
    ):
    load_dotenv(".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )

    # Embedding function initialization
    embedding_function = OllamaEmbeddings(
        model="bge-m3:567m",
    )

    text_splitter = TokenTextSplitter(chunk_size=1200)
    chunks = [Document(text) for text in text_splitter.split_text(context)]

    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function
    )

    retriever = vectorstore.as_retriever()

    content_data = "\n".join([doc.page_content for doc in retriever.invoke(input)])
    
    prompt_template = PROMPTS["naive_rag_response"]
    final_query = prompt_template.format(
        history="",
        content_data=content_data,
        response_type="",
        user_prompt=""
    )

    answer = llm.invoke(final_query)

    return answer
