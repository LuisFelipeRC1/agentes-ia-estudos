"""
exemplo_uso.py - Demonstração completa do agente de IA
========================================================
Este script mostra como usar cada componente do projeto:
1. O agente ReAct com ferramentas externas
2. O pipeline de RAG sobre documentos locais
3. Uso isolado das ferramentas

Execute com:
    python exemplos/exemplo_uso.py

Pré-requisitos:
    1. Instale as dependências: pip install -r requirements.txt
    2. Configure o arquivo .env com suas chaves de API
       (copie .env.example para .env e preencha as chaves)
"""

import sys
import os

# Adiciona o diretório raiz ao path para que os imports funcionem
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agente_ia.ferramentas import buscar_wikipedia, calcular


def demo_ferramentas_isoladas() -> None:
    """Demonstra o uso das ferramentas sem o agente."""
    print("=" * 60)
    print("DEMO 1: Ferramentas isoladas")
    print("=" * 60)

    # Calculadora
    print("\n--- Calculadora ---")
    expressoes = ["2 + 3 * 4", "100 / 4", "2 ** 10", "10 % 3"]
    for expr in expressoes:
        resultado = calcular.invoke({"expressao": expr})
        print(f"  {expr} = {resultado}")

    # Wikipedia
    print("\n--- Wikipedia ---")
    topicos = ["Python (linguagem de programação)", "Inteligência artificial"]
    for topico in topicos:
        print(f"\nBuscando: {topico}")
        resultado = buscar_wikipedia.invoke({"topico": topico})
        # Mostra apenas as primeiras 3 linhas
        linhas = resultado.split("\n")[:3]
        print("\n".join(f"  {l}" for l in linhas if l))


def demo_agente_react() -> None:
    """
    Demonstra o agente ReAct respondendo perguntas que requerem ferramentas.

    Nota: requer LLM configurado via .env
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Agente ReAct com ferramentas")
    print("=" * 60)

    try:
        from agente_ia.agente import AgenteIA

        # Cria o agente com verbose=True para ver o raciocínio interno
        agente = AgenteIA(verbose=True)

        print(f"\nFerramentas disponíveis: {agente.listar_ferramentas()}")
        print("\n--- Iniciando conversa ---\n")

        perguntas = [
            "Quanto é 15% de 480?",
            "Me explique brevemente o que é machine learning.",
        ]

        for pergunta in perguntas:
            print(f"Você: {pergunta}")
            resposta = agente.perguntar(pergunta)
            print(f"Agente: {resposta}\n")
            print("-" * 40)

    except Exception as e:
        print(f"\n[AVISO] Demo do agente ReAct não disponível: {e}")
        print("Configure as chaves de API no arquivo .env para usar o agente.")


def demo_rag() -> None:
    """
    Demonstra o pipeline RAG respondendo perguntas sobre documentos locais.

    Nota: requer LLM e modelo de embeddings configurados via .env
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Pipeline RAG")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        from agente_ia.agente import criar_llm
        from agente_ia.rag import PipelineRAG

        llm = criar_llm()
        embeddings = OpenAIEmbeddings()

        rag = PipelineRAG(llm=llm, embeddings=embeddings)

        print("\nIndexando documentos...")
        total = rag.indexar_documentos()
        print(f"Indexados {total} trechos de texto.")

        perguntas = [
            "O que é o padrão ReAct?",
            "Qual a diferença entre LangChain e LlamaIndex?",
            "O que são embeddings?",
        ]

        for pergunta in perguntas:
            print(f"\nPergunta: {pergunta}")
            resposta = rag.perguntar(pergunta)
            print(f"Resposta: {resposta}")
            print("-" * 40)

    except Exception as e:
        print(f"\n[AVISO] Demo do RAG não disponível: {e}")
        print("Configure as chaves de API no arquivo .env para usar o RAG.")


if __name__ == "__main__":
    print("Agente de IA - Demonstração de Uso")
    print("Projeto educacional: agentes-ia-estudos\n")

    demo_ferramentas_isoladas()
    demo_agente_react()
    demo_rag()

    print("\n" + "=" * 60)
    print("Demonstração concluída!")
    print("=" * 60)
