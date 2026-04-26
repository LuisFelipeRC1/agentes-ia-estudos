"""
prompts.py - Engenharia de Prompts (Prompt Engineering)
=========================================================
Prompt engineering é a arte de criar instruções claras e estruturadas para
guiar um LLM (Large Language Model) a produzir respostas de alta qualidade.

Boas práticas aplicadas aqui:
1. **Papel claro** (role): diga ao modelo quem ele é e qual é seu objetivo.
2. **Contexto estruturado**: separe o contexto do utilizador da instrução.
3. **Formato de saída**: indique claramente o formato esperado da resposta.
4. **Exemplos** (few-shot): quando necessário, forneça exemplos de
   entrada → saída para guiar o modelo.
5. **Restrições explícitas**: diga o que o modelo NÃO deve fazer.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ---------------------------------------------------------------------------
# Prompt do sistema para o agente ReAct
# ---------------------------------------------------------------------------
# O padrão ReAct (Reasoning + Acting) instrui o modelo a:
#   1. Pensar (Thought) - raciocinar sobre o problema
#   2. Agir (Action)    - escolher e chamar uma ferramenta
#   3. Observar (Observation) - analisar o resultado da ferramenta
#   4. Repetir até ter a resposta final

SYSTEM_PROMPT_AGENTE = """Você é um assistente de inteligência artificial especializado \
e prestativo.

Você tem acesso a ferramentas externas para ajudar nas suas respostas. \
Sempre que possível, use as ferramentas para obter informações atualizadas \
e precisas antes de responder.

Diretrizes de comportamento:
- Seja objetivo e direto nas respostas.
- Cite a fonte ou ferramenta utilizada quando relevante.
- Se não souber a resposta e não houver ferramenta disponível, diga isso \
claramente em vez de inventar informações.
- Responda sempre em português (Brasil), a menos que o usuário escreva em \
outro idioma.
"""

# ChatPromptTemplate organiza a conversa em turnos: sistema → histórico → usuário
PROMPT_AGENTE = ChatPromptTemplate.from_messages(
    [
        # Turno 1: instrução global de comportamento (sempre presente)
        ("system", SYSTEM_PROMPT_AGENTE),
        # Turno 2: histórico dinâmico de mensagens (memória de curto prazo)
        # MessagesPlaceholder é substituído pela lista de mensagens anteriores
        MessagesPlaceholder(variable_name="chat_history"),
        # Turno 3: mensagem atual do usuário
        ("human", "{input}"),
        # Turno 4: espaço reservado para o scratchpad do agente ReAct
        # (pensamentos intermediários e chamadas de ferramentas)
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# ---------------------------------------------------------------------------
# Prompt para o pipeline de RAG
# ---------------------------------------------------------------------------
# Na RAG, o contexto recuperado dos documentos é injetado diretamente no
# prompt para que o LLM possa usar informações que não estão em seu
# treinamento original.

SYSTEM_PROMPT_RAG = """Você é um assistente que responde perguntas com base \
estritamente nos trechos de documentos fornecidos abaixo.

Regras importantes:
- Use APENAS as informações dos trechos abaixo para responder.
- Se a resposta não estiver nos trechos, diga: \
"Não encontrei essa informação nos documentos disponíveis."
- Cite de qual trecho você extraiu a informação (ex: [Trecho 1]).
- Não invente informações que não estejam nos trechos.
"""

PROMPT_RAG = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_RAG),
        (
            "human",
            "Trechos dos documentos:\n{context}\n\nPergunta: {question}",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Prompt para resumo de documentos
# ---------------------------------------------------------------------------

PROMPT_RESUMO = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é especialista em síntese de textos. Crie resumos claros, "
            "objetivos e bem estruturados.",
        ),
        (
            "human",
            "Resuma o texto abaixo em no máximo {max_palavras} palavras, "
            "destacando os pontos mais importantes:\n\n{texto}",
        ),
    ]
)
