from langchain_postgres.vectorstores import PGVector
from config import get_embeddings, DATABASE_URL, COLLECTION_NAME

def get_vector_store():
    embeddings = get_embeddings()
    db_url_clean = DATABASE_URL.strip("'").strip('"')
    collection_name_clean = COLLECTION_NAME.strip("'").strip('"')

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name_clean,
        connection=db_url_clean,
        use_jsonb=True
    )
    return vector_store

def search_documents(query: str, k: int = 10):
    vector_store = get_vector_store()
    # Busca por similaridade retornando score
    results = vector_store.similarity_search_with_score(query, k=k)
    return results