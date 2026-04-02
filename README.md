# Desafio MBA Engenharia de Software com IA - Full Cycle

Este projeto consiste em um sistema de Ingestão e Busca Semântica (RAG) utilizando LangChain, Banco de Dados Vetorial (PostgreSQL com pgVector) e suporte dinâmico a LLMs da OpenAI e Google Gemini.

## Pré-requisitos

1. **Docker e Docker Compose** instalados (para subir o banco de dados pgVector).
2. **Python 3.10+** (recomendado `venv`).
3. Chaves de API das plataformas de IA:
   - OpenAI API Key
   - Google API Key

## Como Executar

### 1. Preparar o Ambiente Virtual

Crie o ambiente virtual do Python e instale as bibliotecas necessárias:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar Variáveis de Ambiente

Copie o modelo de arquivo de ambiente e preencha as variáveis com suas credenciais:
```bash
cp .env.example .env
```
* **Nota**: Preencha as chaves (`OPENAI_API_KEY` e/ou `GOOGLE_API_KEY`). A configuração global é controlada pela tag `LLM_PROVIDER` (sendo `openai` a aplicação padrão).

### 3. Rodar o Banco de Dados Vetorial

Suba o serviço do banco pgVector em background:
```bash
docker compose up -d
```

### 4. Executar Ingestão do PDF

Realize a leitura do arquivo PDF apontado no `.env` (padrão `document.pdf`) para criar as frações e salvar o embedding de cada fragmento no Vector Store:
```bash
python src/ingest.py
```
*(Certifique-se de aguardar a mensagem de sucesso ao final do carregamento e gravação).*

### 5. Conversar com seus Dados Interativamente

Assim que a ingestão terminar, inicie o CLI para testar a busca:
```bash
python src/chat.py
```

> No terminal, escreva suas perguntas. O sistema restringirá suas respostas baseando-se explicitamente apenas nos tópicos cobertos pelo documento PDF indexado no banco vetorial.