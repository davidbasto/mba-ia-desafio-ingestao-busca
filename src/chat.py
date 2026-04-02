import sys
from langchain_core.prompts import PromptTemplate
from search import search_documents
from config import get_llm

prompt_template = """CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def main():
    print("Inicializando chat...", flush=True)
    prompt = PromptTemplate.from_template(prompt_template)
    llm = get_llm()
    chain = prompt | llm

    print("\n" + "="*50)
    print("Chat Iniciado. Pressione Ctrl+C para sair.")
    print("="*50 + "\n")

    while True:
        try:
            query = input("Faça sua pergunta:\n\nPERGUNTA: ").strip()
            if not query:
                continue
            
            # Buscar os 10 resultados mais relevantes (k=10) no banco vetorial.
            results = search_documents(query, k=10)
            
            # Montar a string de contexto
            context_texts = []
            for doc, score in results:
                context_texts.append(doc.page_content)

            context_str = "\n".join(context_texts)
            
            # Executar a rede (Chamar a LLM via cadeia)
            response = chain.invoke({
                "context": context_str,
                "question": query
            })
            
            print(f"RESPOSTA: {response.content}\n")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nEncerrando o chat...")
            sys.exit(0)
        except Exception as e:
            print(f"\nOcorreu um erro: {e}\n")

if __name__ == "__main__":
    main()