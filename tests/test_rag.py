"""
test_rag.py - Testes do pipeline de RAG
=========================================
Estes testes validam a lógica do pipeline RAG usando mocks para simular
o LLM, o modelo de embeddings e o banco vetorial FAISS.

O que testamos:
1. Inicialização correta do pipeline
2. Indexação de documentos
3. Formatação dos documentos recuperados
4. Geração de respostas baseadas em contexto
5. Tratamento de erros (diretório não encontrado, sem documentos, etc.)
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agente_ia.rag import PipelineRAG


def criar_rag_mockado(tmp_dir: str = "dados/documentos") -> PipelineRAG:
    """
    Helper: cria uma instância de PipelineRAG com dependências mockadas.

    Args:
        tmp_dir: Diretório de documentos a usar.

    Returns:
        Instância de PipelineRAG com LLM e embeddings mockados.
    """
    llm_mock = MagicMock()
    embeddings_mock = MagicMock()
    return PipelineRAG(llm=llm_mock, embeddings=embeddings_mock, docs_dir=tmp_dir)


class TestPipelineRAGInicializacao:
    """Testes de inicialização do PipelineRAG."""

    def test_inicializacao_com_valores_padrao(self):
        """Testa que o pipeline inicializa com valores padrão corretos."""
        rag = criar_rag_mockado()

        assert rag.vectorstore is None
        assert rag.retriever is None
        assert rag.top_k == 3  # valor padrão de RAG_TOP_K

    def test_inicializacao_com_valores_customizados(self):
        """Testa inicialização com parâmetros customizados."""
        llm_mock = MagicMock()
        embeddings_mock = MagicMock()

        rag = PipelineRAG(
            llm=llm_mock,
            embeddings=embeddings_mock,
            docs_dir="/custom/dir",
            top_k=5,
        )

        assert rag.docs_dir == "/custom/dir"
        assert rag.top_k == 5


class TestPipelineRAGIndexacao:
    """Testes do método indexar_documentos."""

    def test_diretorio_inexistente_lanca_erro(self):
        """Testa que diretório não existente lança FileNotFoundError."""
        rag = criar_rag_mockado(tmp_dir="/diretorio/que/nao/existe")

        with pytest.raises(FileNotFoundError, match="não encontrado"):
            rag.indexar_documentos()

    def test_diretorio_vazio_lanca_erro(self):
        """Testa que diretório sem documentos .txt lança ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rag = criar_rag_mockado(tmp_dir=tmp_dir)

            with pytest.raises(ValueError, match="Nenhum documento"):
                rag.indexar_documentos()

    @patch("agente_ia.rag.FAISS")
    @patch("agente_ia.rag.DirectoryLoader")
    @patch("agente_ia.rag.RecursiveCharacterTextSplitter")
    def test_indexacao_bem_sucedida(
        self, mock_splitter_cls, mock_loader_cls, mock_faiss_cls
    ):
        """Testa indexação bem-sucedida com documentos mockados."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cria um arquivo de texto simulado
            arquivo_txt = Path(tmp_dir) / "doc.txt"
            arquivo_txt.write_text("Conteúdo de teste para indexação.", encoding="utf-8")

            # Configura os mocks
            doc_mock = MagicMock()
            doc_mock.page_content = "Conteúdo de teste"

            mock_loader = MagicMock()
            mock_loader.load.return_value = [doc_mock]
            mock_loader_cls.return_value = mock_loader

            mock_splitter = MagicMock()
            mock_splitter.split_documents.return_value = [doc_mock, doc_mock]
            mock_splitter_cls.return_value = mock_splitter

            mock_vectorstore = MagicMock()
            mock_vectorstore.index.ntotal = 2
            mock_faiss_cls.from_documents.return_value = mock_vectorstore

            rag = criar_rag_mockado(tmp_dir=tmp_dir)
            total = rag.indexar_documentos()

            assert total == 2  # 2 trechos mockados
            assert rag.vectorstore is not None
            assert rag.retriever is not None

    @patch("agente_ia.rag.FAISS")
    def test_carrega_indice_existente(self, mock_faiss_cls):
        """Testa que índice existente é carregado sem reindexar."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Simula existência do índice
            indice_path = Path(tmp_dir) / "indice_faiss"
            indice_path.mkdir()

            mock_vectorstore = MagicMock()
            mock_vectorstore.index.ntotal = 5
            mock_faiss_cls.load_local.return_value = mock_vectorstore

            rag = criar_rag_mockado(tmp_dir=tmp_dir)
            total = rag.indexar_documentos()

            assert total == 5
            mock_faiss_cls.load_local.assert_called_once()
            # Não deve chamar from_documents (reindexação)
            mock_faiss_cls.from_documents.assert_not_called()


class TestPipelineRAGConsulta:
    """Testes dos métodos de consulta do PipelineRAG."""

    def test_perguntar_sem_indexar_lanca_erro(self):
        """Testa que perguntar() sem indexar lança RuntimeError."""
        rag = criar_rag_mockado()

        with pytest.raises(RuntimeError, match="ainda não indexados"):
            rag.perguntar("Qual é a resposta?")

    def test_buscar_trechos_sem_indexar_lanca_erro(self):
        """Testa que buscar_trechos_similares() sem indexar lança RuntimeError."""
        rag = criar_rag_mockado()

        with pytest.raises(RuntimeError, match="ainda não indexados"):
            rag.buscar_trechos_similares("consulta teste")

    def test_formatar_documentos(self):
        """Testa a formatação dos documentos recuperados."""
        rag = criar_rag_mockado()

        # Cria documentos mock
        doc1 = MagicMock()
        doc1.page_content = "Primeiro trecho de texto."
        doc1.metadata = {"source": "/caminho/para/arquivo1.txt"}

        doc2 = MagicMock()
        doc2.page_content = "Segundo trecho de texto."
        doc2.metadata = {"source": "/caminho/para/arquivo2.txt"}

        resultado = rag._formatar_documentos([doc1, doc2])

        assert "[Trecho 1]" in resultado
        assert "[Trecho 2]" in resultado
        assert "arquivo1.txt" in resultado
        assert "arquivo2.txt" in resultado
        assert "Primeiro trecho" in resultado
        assert "Segundo trecho" in resultado

    def test_formatar_documentos_sem_fonte(self):
        """Testa formatação quando metadado de fonte está ausente."""
        rag = criar_rag_mockado()

        doc = MagicMock()
        doc.page_content = "Trecho sem fonte."
        doc.metadata = {}  # sem chave 'source'

        resultado = rag._formatar_documentos([doc])

        assert "[Trecho 1]" in resultado
        assert "Trecho sem fonte." in resultado

    def test_buscar_trechos_similares_com_vectorstore(self):
        """Testa busca de trechos quando vectorstore está configurado."""
        rag = criar_rag_mockado()

        doc_mock = MagicMock()
        rag.vectorstore = MagicMock()
        rag.vectorstore.similarity_search.return_value = [doc_mock]

        resultado = rag.buscar_trechos_similares("consulta teste")

        assert resultado == [doc_mock]
        rag.vectorstore.similarity_search.assert_called_once_with(
            "consulta teste", k=rag.top_k
        )
