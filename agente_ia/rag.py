"""
rag.py - Pipeline de RAG (Retrieval-Augmented Generation)
===========================================================
RAG é uma técnica que combina busca de informação (Retrieval) com geração de
texto (Generation) para produzir respostas baseadas em documentos específicos,
não apenas no conhecimento de treinamento do LLM.

Fluxo do pipeline RAG:
  1. INDEXAÇÃO (feita uma vez):
     a. Carregar documentos (PDF, TXT, etc.)
     b. Dividir em trechos menores (chunks)
     c. Gerar embeddings (vetores numéricos) para cada trecho
     d. Armazenar os vetores em um banco vetorial (ex: FAISS)

  2. CONSULTA (feita a cada pergunta):
     a. Converter a pergunta em um embedding
     b. Buscar os trechos mais similares no banco vetorial (busca semântica)
     c. Injetar esses trechos no prompt do LLM como contexto
     d. O LLM gera a resposta baseada APENAS no contexto fornecido

Por que RAG?
- Permite ao LLM "saber" sobre documentos privados ou recentes.
- Reduz alucinações ao ancorar as respostas em fontes reais.
- Mais econômico que re-treinar o modelo.
"""

import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agente_ia.config import DOCS_DIR, RAG_TOP_K
from agente_ia.prompts import PROMPT_RAG


class PipelineRAG:
    """
    Pipeline completo de RAG para responder perguntas sobre documentos locais.

    Attributes:
        llm: Modelo de linguagem (LLM) para geração das respostas.
        embeddings: Modelo de embeddings para vetorizar texto.
        vectorstore: Banco vetorial que armazena os documentos indexados.
        retriever: Componente que busca trechos relevantes no vectorstore.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        docs_dir: str = DOCS_DIR,
        top_k: int = RAG_TOP_K,
    ) -> None:
        """
        Inicializa o pipeline RAG.

        Args:
            llm: Instância do LLM para geração de respostas.
            embeddings: Modelo de embeddings (para vetorizar textos).
            docs_dir: Caminho para o diretório com os documentos.
            top_k: Número de trechos a recuperar por consulta.
        """
        self.llm = llm
        self.embeddings = embeddings
        self.docs_dir = docs_dir
        self.top_k = top_k
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None

    def indexar_documentos(self, forcar_reindexacao: bool = False) -> int:
        """
        Carrega, divide e indexa os documentos do diretório configurado.

        O índice FAISS é salvo em disco para evitar reindexações desnecessárias.
        Use forcar_reindexacao=True para reindexar mesmo que o índice já exista.

        Args:
            forcar_reindexacao: Se True, recria o índice mesmo que já exista.

        Returns:
            Número de trechos indexados.

        Raises:
            FileNotFoundError: Se o diretório de documentos não existir.
            ValueError: Se nenhum documento for encontrado no diretório.
        """
        indice_path = os.path.join(self.docs_dir, "indice_faiss")

        # Carrega índice existente se disponível (economiza tempo e custo de API)
        if not forcar_reindexacao and os.path.exists(indice_path):
            self.vectorstore = FAISS.load_local(
                indice_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.top_k}
            )
            return self.vectorstore.index.ntotal

        # Valida se o diretório de documentos existe
        if not Path(self.docs_dir).exists():
            raise FileNotFoundError(
                f"Diretório de documentos não encontrado: '{self.docs_dir}'. "
                "Crie o diretório e adicione arquivos .txt para indexar."
            )

        # 1. CARREGAMENTO: lê todos os arquivos .txt do diretório
        loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        documentos = loader.load()

        if not documentos:
            raise ValueError(
                f"Nenhum documento .txt encontrado em '{self.docs_dir}'. "
                "Adicione arquivos de texto para indexar."
            )

        # 2. DIVISÃO: fragmenta os documentos em trechos menores
        # chunk_size: tamanho máximo de cada trecho em caracteres
        # chunk_overlap: sobreposição entre trechos para não perder contexto
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "],
        )
        trechos = splitter.split_documents(documentos)

        if not trechos:
            raise ValueError("Os documentos estão vazios após a divisão em trechos.")

        # 3. EMBEDDINGS + INDEXAÇÃO: vetoriza os trechos e cria o índice FAISS
        # FAISS (Facebook AI Similarity Search) é uma biblioteca eficiente
        # para busca de vetores similares
        self.vectorstore = FAISS.from_documents(trechos, self.embeddings)

        # Salva o índice em disco para reutilização futura
        self.vectorstore.save_local(indice_path)

        # Cria o retriever com configuração de quantos trechos retornar
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )

        return len(trechos)

    def _formatar_documentos(self, docs: list) -> str:
        """
        Formata a lista de documentos recuperados em uma string numerada.

        Args:
            docs: Lista de objetos Document do LangChain.

        Returns:
            String formatada com os trechos numerados.
        """
        trechos_formatados = []
        for i, doc in enumerate(docs, start=1):
            fonte = doc.metadata.get("source", "desconhecida")
            nome_arquivo = Path(fonte).name if fonte else "desconhecida"
            trechos_formatados.append(
                f"[Trecho {i}] (fonte: {nome_arquivo})\n{doc.page_content}"
            )
        return "\n\n".join(trechos_formatados)

    def perguntar(self, pergunta: str) -> str:
        """
        Responde a uma pergunta usando RAG.

        Fluxo:
        1. Converte a pergunta em embedding
        2. Busca os trechos mais similares no índice
        3. Injeta os trechos no prompt
        4. O LLM gera a resposta baseada nos trechos

        Args:
            pergunta: A pergunta do usuário em linguagem natural.

        Returns:
            Resposta gerada pelo LLM com base nos documentos.

        Raises:
            RuntimeError: Se os documentos ainda não foram indexados.
        """
        if self.retriever is None:
            raise RuntimeError(
                "Documentos ainda não indexados. "
                "Chame indexar_documentos() antes de fazer perguntas."
            )

        # Cadeia RAG usando LCEL (LangChain Expression Language)
        # O pipe (|) encadeia os componentes: retriever → prompt → llm → parser
        cadeia_rag = (
            {
                # Recupera os documentos relevantes e os formata como contexto
                "context": self.retriever | self._formatar_documentos,
                # Passa a pergunta original sem alterações
                "question": RunnablePassthrough(),
            }
            | PROMPT_RAG        # Monta o prompt com contexto + pergunta
            | self.llm          # Envia para o LLM gerar a resposta
            | StrOutputParser() # Extrai o texto da resposta do LLM
        )

        return cadeia_rag.invoke(pergunta)

    def buscar_trechos_similares(self, consulta: str) -> list:
        """
        Retorna os trechos de documentos mais similares à consulta,
        sem gerar uma resposta com o LLM. Útil para depuração e inspeção.

        Args:
            consulta: Texto da consulta em linguagem natural.

        Returns:
            Lista de objetos Document com os trechos mais relevantes.

        Raises:
            RuntimeError: Se os documentos ainda não foram indexados.
        """
        if self.vectorstore is None:
            raise RuntimeError(
                "Documentos ainda não indexados. "
                "Chame indexar_documentos() antes de buscar."
            )
        return self.vectorstore.similarity_search(consulta, k=self.top_k)
