import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
if not LLM_PROVIDER:
    LLM_PROVIDER = "openai"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
PDF_PATH = os.getenv("PDF_PATH", "document.pdf")

def get_embeddings():
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model_name = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001").strip("'").strip('"')
        return GoogleGenerativeAIEmbeddings(model=model_name)
    else:
        from langchain_openai import OpenAIEmbeddings
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip("'").strip('"')
        return OpenAIEmbeddings(model=model_name)

def get_llm():
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm_model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite").strip("'").strip('"')
        return ChatGoogleGenerativeAI(model=llm_model)
    else:
        from langchain_openai import ChatOpenAI
        llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano").strip("'").strip('"')
        return ChatOpenAI(model=llm_model)
