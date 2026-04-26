"""
agente.py - Agente ReAct principal
====================================
O padrão ReAct (Reasoning + Acting) é uma arquitetura para agentes de IA que
combina raciocínio em linguagem natural com execução de ações externas.

Ciclo ReAct:
  Thought (Pensamento) → Action (Ação) → Observation (Observação) → ...
  ... → Thought → Final Answer (Resposta Final)

Exemplo visual do ciclo:
  Usuário: "Qual é a temperatura em São Paulo hoje?"

  Thought: Preciso consultar o clima atual de São Paulo.
  Action: buscar_clima("São Paulo")
  Observation: Clima em São Paulo: Nublado. Temperatura: 22.3°C

  Thought: Tenho a informação necessária para responder.
  Final Answer: A temperatura atual em São Paulo é 22.3°C com tempo nublado.

Por que usar agentes ao invés de prompts simples?
- Podem usar ferramentas para obter informações em tempo real.
- Raciocinam em múltiplos passos para resolver problemas complexos.
- São mais flexíveis: você adiciona/remove ferramentas sem mudar a arquitetura.
"""

from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agente_ia.config import (
    GOOGLE_API_KEY,
    LLM_PROVIDER,
    MODEL_NAME,
    OPENAI_API_KEY,
    TEMPERATURE,
)
from agente_ia.ferramentas import FERRAMENTAS_DISPONIVEIS
from agente_ia.prompts import PROMPT_AGENTE


def criar_llm() -> BaseChatModel:
    """
    Cria e retorna a instância do LLM conforme o provedor configurado.

    O provedor é definido pela variável de ambiente LLM_PROVIDER.
    - "openai"  → usa langchain_openai.ChatOpenAI
    - "gemini"  → usa langchain_google_genai.ChatGoogleGenerativeAI

    Returns:
        Instância do LLM configurado.

    Raises:
        ValueError: Se o provedor configurado não for suportado.
        ImportError: Se o pacote do provedor não estiver instalado.
    """
    if LLM_PROVIDER == "openai":
        # Importação lazy: só carrega se o provedor for OpenAI
        from langchain_openai import ChatOpenAI  # type: ignore

        return ChatOpenAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY or None,
        )

    if LLM_PROVIDER == "gemini":
        # Importação lazy: só carrega se o provedor for Google Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

        return ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            google_api_key=GOOGLE_API_KEY or None,
        )

    raise ValueError(
        f"Provedor de LLM não suportado: '{LLM_PROVIDER}'. "
        "Use 'openai' ou 'gemini'."
    )


class AgenteIA:
    """
    Agente de Inteligência Artificial com padrão ReAct.

    Características:
    - Raciocínio em múltiplos passos (chain-of-thought)
    - Acesso a ferramentas externas (clima, calculadora, Wikipedia)
    - Memória de conversa (histórico de mensagens)
    - Suporte a múltiplos provedores de LLM (OpenAI e Google Gemini)

    Attributes:
        llm: Modelo de linguagem utilizado pelo agente.
        ferramentas: Lista de ferramentas disponíveis.
        historico: Histórico de mensagens da conversa atual.
        executor: Executor do agente LangChain que gerencia o ciclo ReAct.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        ferramentas: Optional[list] = None,
        verbose: bool = False,
    ) -> None:
        """
        Inicializa o agente ReAct.

        Args:
            llm: Instância do LLM. Se None, cria um com base nas configurações.
            ferramentas: Lista de ferramentas. Se None, usa todas as disponíveis.
            verbose: Se True, exibe o raciocínio interno do agente (útil para
                     depuração e aprendizado).
        """
        # Usa o LLM fornecido ou cria um novo com base nas configurações
        self.llm: BaseChatModel = llm or criar_llm()

        # Usa as ferramentas fornecidas ou todas as disponíveis
        self.ferramentas: list = ferramentas or FERRAMENTAS_DISPONIVEIS

        # Histórico de conversa: mantém o contexto entre perguntas
        self.historico: list[BaseMessage] = []

        # Cria o agente usando a função helper do LangChain
        # create_tool_calling_agent conecta o LLM com as ferramentas
        agente = create_tool_calling_agent(
            llm=self.llm,
            tools=self.ferramentas,
            prompt=PROMPT_AGENTE,
        )

        # AgentExecutor orquestra o ciclo ReAct:
        # chama o agente → executa ferramenta → volta ao agente → ...
        self.executor = AgentExecutor(
            agent=agente,
            tools=self.ferramentas,
            verbose=verbose,              # exibe pensamentos intermediários
            handle_parsing_errors=True,   # não quebra em erros de parsing
            max_iterations=10,            # evita loops infinitos
        )

    def perguntar(self, pergunta: str) -> str:
        """
        Envia uma pergunta ao agente e retorna a resposta.

        O agente pode usar ferramentas zero ou mais vezes antes de responder.
        O histórico da conversa é mantido automaticamente.

        Args:
            pergunta: Texto da pergunta do usuário.

        Returns:
            Resposta gerada pelo agente como string.
        """
        # Invoca o executor com a pergunta e uma cópia do histórico acumulado.
        # Passamos uma cópia (list(self.historico)) para que o executor não
        # receba uma referência mutável que poderia mudar durante a execução.
        resultado = self.executor.invoke(
            {
                "input": pergunta,
                "chat_history": list(self.historico),
            }
        )

        resposta = resultado["output"]

        # Atualiza o histórico com este turno de conversa
        # HumanMessage = mensagem do usuário
        # AIMessage = mensagem do assistente (agente)
        self.historico.extend(
            [
                HumanMessage(content=pergunta),
                AIMessage(content=resposta),
            ]
        )

        return resposta

    def limpar_historico(self) -> None:
        """
        Limpa o histórico de conversa, iniciando uma nova sessão.
        Use quando quiser começar uma conversa do zero.
        """
        self.historico = []

    def listar_ferramentas(self) -> list[str]:
        """
        Retorna os nomes das ferramentas disponíveis para o agente.

        Returns:
            Lista com os nomes das ferramentas.
        """
        return [ferramenta.name for ferramenta in self.ferramentas]
