"""
config.py - Configurações globais do agente
============================================
Este módulo centraliza todas as configurações do projeto, lendo-as de variáveis
de ambiente definidas no arquivo .env (nunca versione chaves reais!).

Por que usar variáveis de ambiente?
- Segurança: chaves de API ficam fora do código-fonte.
- Portabilidade: cada desenvolvedor/servidor usa suas próprias credenciais.
- Praticidade: um único ponto de mudança para toda a aplicação.
"""

import os
from dotenv import load_dotenv

# Carrega as variáveis definidas em .env para o ambiente do processo
load_dotenv()


# ---------------------------------------------------------------------------
# Provedor e modelo de LLM
# ---------------------------------------------------------------------------

# Qual empresa de LLM usar: "openai" ou "gemini"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")

# Nome do modelo; pode ser sobrescrito via .env
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Temperatura controla a aleatoriedade das respostas.
# 0.0 = respostas mais determinísticas (bom para tarefas que exigem precisão)
# 1.0 = respostas mais criativas/variadas
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

# Chaves de API (lidas do ambiente, nunca hardcoded)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")


# ---------------------------------------------------------------------------
# Configurações de RAG (Retrieval-Augmented Generation)
# ---------------------------------------------------------------------------

# Diretório local onde estão os documentos a indexar
DOCS_DIR: str = os.getenv("DOCS_DIR", "dados/documentos")

# Quantos trechos relevantes retornar na busca vetorial
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))


# ---------------------------------------------------------------------------
# Ferramentas externas
# ---------------------------------------------------------------------------

# Chave para a API de clima (OpenWeatherMap)
OPENWEATHERMAP_API_KEY: str = os.getenv("OPENWEATHERMAP_API_KEY", "")
