# 🤖 Agentes de IA — Estudos Práticos

Projeto educacional em Python que implementa um **agente de Inteligência Artificial** com padrão **ReAct** (Reasoning + Acting), **RAG** (Retrieval-Augmented Generation), integração com **APIs externas** e **prompt engineering** estruturado.

Desenvolvido como estudo prático das tecnologias exigidas no mercado de desenvolvimento de agentes de IA.

---

## 📋 Sumário

- [Estrutura do Projeto](#estrutura-do-projeto)
- [Conceitos Abordados](#conceitos-abordados)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Como Usar](#como-usar)
- [Módulos — Explicação Didática](#módulos--explicação-didática)
- [Testes](#testes)
- [Referências](#referências)

---

## 📁 Estrutura do Projeto

```
agentes-ia-estudos/
├── agente_ia/                  # Pacote principal
│   ├── __init__.py             # Inicialização do pacote
│   ├── config.py               # Configurações via variáveis de ambiente
│   ├── prompts.py              # Templates de prompts (Prompt Engineering)
│   ├── ferramentas.py          # Ferramentas externas (Tools)
│   ├── rag.py                  # Pipeline de RAG
│   └── agente.py               # Agente ReAct principal
├── dados/
│   └── documentos/             # Documentos .txt para indexação RAG
│       └── ia_conceitos.txt    # Documento de exemplo
├── exemplos/
│   └── exemplo_uso.py          # Script de demonstração completa
├── tests/                      # Testes unitários com pytest
│   ├── test_agente.py
│   ├── test_ferramentas.py
│   └── test_rag.py
├── .env.example                # Exemplo de variáveis de ambiente
├── requirements.txt            # Dependências Python
└── README.md
```

---

## 🧠 Conceitos Abordados

### 1. LLMs (Large Language Models)
Modelos de linguagem de grande escala como **GPT-4o** (OpenAI) e **Gemini** (Google). O projeto suporta ambos, configuráveis via variável de ambiente.

### 2. Prompt Engineering
A arte de criar instruções claras para guiar o LLM. Técnicas usadas:
- **System prompt**: instrução global de comportamento
- **Few-shot**: o modelo aprende com exemplos
- **Chain-of-thought**: o modelo "pensa em voz alta" antes de responder
- **Role prompting**: o modelo assume um papel específico

### 3. Padrão ReAct (Reasoning + Acting)
Arquitetura em que o agente alterna entre **pensar** (Thought) e **agir** (Action):

```
Pergunta → Thought → Action (ferramenta) → Observation → Thought → ... → Resposta Final
```

### 4. Ferramentas (Tools)
Funções que o agente pode invocar para obter informações externas:
- 🌡️ **buscar_clima** — temperatura atual via API REST
- 🔢 **calcular** — avalia expressões matemáticas com segurança
- 📚 **buscar_wikipedia** — resumos de artigos da Wikipedia

### 5. RAG (Retrieval-Augmented Generation)
Pipeline que permite ao LLM responder sobre documentos privados:

```
Documentos → Embeddings → Índice FAISS → Busca Semântica → Contexto → LLM → Resposta
```

### 6. LangChain
Framework que une todos os componentes acima de forma modular e declarativa.

---

## ✅ Pré-requisitos

- Python 3.10 ou superior
- Chave de API da [OpenAI](https://platform.openai.com) **ou** do [Google Gemini](https://aistudio.google.com)
- (Opcional) Chave de API do [OpenWeatherMap](https://openweathermap.org/api) para a ferramenta de clima

---

## 🔧 Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/LuisFelipeRC1/agentes-ia-estudos.git
cd agentes-ia-estudos

# 2. Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## ⚙️ Configuração

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o .env com suas chaves
nano .env   # ou use qualquer editor
```

Variáveis principais no `.env`:

| Variável | Descrição | Padrão |
|---|---|---|
| `LLM_PROVIDER` | Provedor do LLM: `openai` ou `gemini` | `openai` |
| `OPENAI_API_KEY` | Chave da API OpenAI | — |
| `GOOGLE_API_KEY` | Chave da API Google Gemini | — |
| `MODEL_NAME` | Modelo a usar (ex: `gpt-4o-mini`) | `gpt-4o-mini` |
| `TEMPERATURE` | Criatividade do modelo (0.0–1.0) | `0.0` |
| `DOCS_DIR` | Diretório dos documentos para RAG | `dados/documentos` |
| `RAG_TOP_K` | Trechos retornados na busca vetorial | `3` |
| `OPENWEATHERMAP_API_KEY` | Chave para ferramenta de clima | — |

---

## 🚀 Como Usar

### Agente ReAct (modo conversacional)

```python
from agente_ia.agente import AgenteIA

# Cria o agente (lê configurações do .env)
agente = AgenteIA(verbose=True)  # verbose=True mostra o raciocínio interno

# Faz perguntas
resposta = agente.perguntar("Quanto é 15% de 480?")
print(resposta)

resposta = agente.perguntar("O que é machine learning?")
print(resposta)

# O agente mantém o histórico da conversa
resposta = agente.perguntar("Me dê mais detalhes sobre o que você acabou de explicar.")
print(resposta)

# Listar ferramentas disponíveis
print(agente.listar_ferramentas())
# ['buscar_clima', 'calcular', 'buscar_wikipedia']

# Iniciar nova conversa
agente.limpar_historico()
```

### Pipeline RAG (perguntas sobre documentos)

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from agente_ia.rag import PipelineRAG

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

rag = PipelineRAG(llm=llm, embeddings=embeddings)

# Indexa os documentos (salvo em disco para reutilização)
total = rag.indexar_documentos()
print(f"Indexados {total} trechos")

# Faz perguntas baseadas nos documentos
resposta = rag.perguntar("O que é o padrão ReAct?")
print(resposta)
```

### Ferramentas isoladas

```python
from agente_ia.ferramentas import calcular, buscar_wikipedia, buscar_clima

# Calculadora segura
print(calcular.invoke({"expressao": "2 ** 10"}))   # "1024"

# Wikipedia
print(buscar_wikipedia.invoke({"topico": "Inteligência artificial"}))

# Clima (requer chave da API)
print(buscar_clima.invoke({"cidade": "São Paulo"}))
```

### Script de demonstração completo

```bash
python exemplos/exemplo_uso.py
```

---

## 📚 Módulos — Explicação Didática

### `config.py` — Configurações
Centraliza todas as configurações lidas de variáveis de ambiente via `python-dotenv`. Evita hardcoding de chaves de API no código.

**Por que isso é importante?** Segurança (chaves fora do código), portabilidade (cada ambiente usa suas credenciais) e manutenibilidade (um único ponto de mudança).

---

### `prompts.py` — Engenharia de Prompts
Define templates reutilizáveis usando `ChatPromptTemplate` do LangChain:

- **`PROMPT_AGENTE`**: prompt para o agente ReAct com system prompt, histórico de conversa (`MessagesPlaceholder`) e scratchpad interno.
- **`PROMPT_RAG`**: prompt que injeta o contexto dos documentos recuperados.
- **`PROMPT_RESUMO`**: prompt para sumarização de textos.

**Técnica aplicada:** separação clara entre instrução de sistema, histórico e input do usuário — evita que o modelo "esqueça" as regras definidas no system prompt.

---

### `ferramentas.py` — Ferramentas Externas
Implementa as ferramentas usando o decorador `@tool` do LangChain:

```python
@tool
def calcular(expressao: str) -> str:
    """Calcula o resultado de uma expressão matemática."""
    ...
```

O docstring é crucial: é ele que o LLM lê para decidir se e quando usar a ferramenta.

**Calculadora segura:** usa `ast.parse` para analisar a expressão matematicamente sem usar `eval()` — evita injeção de código malicioso.

---

### `rag.py` — Pipeline RAG
Implementa o fluxo completo de RAG:

1. **Carregamento** (`DirectoryLoader`): lê arquivos `.txt` do diretório configurado.
2. **Divisão** (`RecursiveCharacterTextSplitter`): fragmenta documentos em trechos de 1000 caracteres com sobreposição de 200 (para não perder contexto nas fronteiras).
3. **Vetorização** (Embeddings): converte cada trecho em um vetor numérico de alta dimensão.
4. **Indexação** (`FAISS`): armazena os vetores em um índice de busca eficiente.
5. **Consulta**: converte a pergunta em vetor → busca os K trechos mais similares → injeta no prompt → LLM responde.

O índice é salvo em disco para não precisar reindexar a cada execução.

---

### `agente.py` — Agente ReAct
Orquestra todos os componentes usando `create_tool_calling_agent` e `AgentExecutor` do LangChain:

- **`criar_llm()`**: factory function que instancia o LLM correto conforme `LLM_PROVIDER`.
- **`AgenteIA`**: classe principal que mantém estado (histórico), cria o executor e expõe a interface pública (`perguntar`, `limpar_historico`, `listar_ferramentas`).

O histórico é passado como cópia a cada chamada para garantir imutabilidade durante a execução.

---

## 🧪 Testes

```bash
# Roda todos os testes
pytest tests/ -v

# Apenas ferramentas
pytest tests/test_ferramentas.py -v

# Apenas agente
pytest tests/test_agente.py -v

# Apenas RAG
pytest tests/test_rag.py -v
```

Os testes usam **mocks** (`unittest.mock`) para simular LLMs, APIs externas e bancos vetoriais, garantindo testes rápidos, determinísticos e sem custo de API.

---

## 🔗 Referências

- [LangChain Docs](https://python.langchain.com/)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [RAG — Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [FAISS — Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [OpenAI API](https://platform.openai.com/docs)
- [Google Gemini API](https://ai.google.dev/)
