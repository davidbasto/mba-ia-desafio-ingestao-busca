import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector

from config import get_embeddings, DATABASE_URL, COLLECTION_NAME, PDF_PATH

def main():
    pdf_path_clean = PDF_PATH.strip("'").strip('"')
    if not os.path.exists(pdf_path_clean):
        print(f"Erro: Arquivo PDF '{pdf_path_clean}' não encontrado.")
        return

    print("Carregando o documento PDF...")
    loader = PyPDFLoader(pdf_path_clean)
    docs = loader.load()

    print("Dividindo o documento em chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    splits = splitter.split_documents(docs)

    print(f"Total de chunks gerados: {len(splits)}")

    print("Inicializando o modelo de embeddings...")
    embeddings = get_embeddings()

    db_url_clean = DATABASE_URL.strip("'").strip('"')
    collection_name_clean = COLLECTION_NAME.strip("'").strip('"')

    print("Conectando ao banco de dados e salvando vetores...")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name_clean,
        connection=db_url_clean,
        use_jsonb=True
    )

    BATCH_SIZE = 500
    total_splits = len(splits)
    print(f"Salvando {total_splits} chunks no banco em lotes de {BATCH_SIZE}...")
    
    for i in range(0, total_splits, BATCH_SIZE):
        batch = splits[i:i + BATCH_SIZE]
        print(f"Processando lote {(i // BATCH_SIZE) + 1} de {(total_splits + BATCH_SIZE - 1) // BATCH_SIZE}...")
        vector_store.add_documents(batch)
        time.sleep(1)
        
    print("Ingestão concluída com sucesso!")

if __name__ == "__main__":
    main()