"""
test_ferramentas.py - Testes das ferramentas externas
======================================================
Estes testes validam o comportamento das ferramentas sem chamar APIs reais,
usando mocks (objetos simulados) para isolar a lógica de negócio.

Por que usar mocks?
- Testes rápidos: não dependem de internet ou chaves de API.
- Determinísticos: sempre produzem o mesmo resultado.
- Isolados: testam apenas o código da ferramenta, não a API externa.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from agente_ia.ferramentas import buscar_clima, buscar_wikipedia, calcular


# ---------------------------------------------------------------------------
# Testes da calculadora
# ---------------------------------------------------------------------------


class TestCalcular:
    """Testes para a ferramenta `calcular`."""

    def test_adicao(self):
        """Testa operação de adição simples."""
        assert calcular.invoke({"expressao": "2 + 3"}) == "5"

    def test_subtracao(self):
        """Testa operação de subtração."""
        assert calcular.invoke({"expressao": "10 - 4"}) == "6"

    def test_multiplicacao(self):
        """Testa operação de multiplicação."""
        assert calcular.invoke({"expressao": "3 * 7"}) == "21"

    def test_divisao(self):
        """Testa operação de divisão com resultado inteiro."""
        assert calcular.invoke({"expressao": "15 / 3"}) == "5"

    def test_divisao_com_decimal(self):
        """Testa divisão com resultado decimal."""
        resultado = calcular.invoke({"expressao": "7 / 2"})
        assert resultado == "3.5"

    def test_potenciacao(self):
        """Testa operação de potenciação."""
        assert calcular.invoke({"expressao": "2 ** 10"}) == "1024"

    def test_modulo(self):
        """Testa operação de módulo (resto da divisão)."""
        assert calcular.invoke({"expressao": "10 % 3"}) == "1"

    def test_expressao_composta(self):
        """Testa expressão com múltiplos operadores e precedência."""
        # 2 + 3 * 4 = 2 + 12 = 14 (multiplicação tem precedência)
        assert calcular.invoke({"expressao": "2 + 3 * 4"}) == "14"

    def test_parenteses(self):
        """Testa uso de parênteses para controlar precedência."""
        # (2 + 3) * 4 = 5 * 4 = 20
        assert calcular.invoke({"expressao": "(2 + 3) * 4"}) == "20"

    def test_numero_negativo(self):
        """Testa operação com número negativo."""
        resultado = calcular.invoke({"expressao": "-5 + 3"})
        assert resultado == "-2"

    def test_divisao_por_zero(self):
        """Testa que divisão por zero retorna mensagem de erro."""
        resultado = calcular.invoke({"expressao": "10 / 0"})
        assert "erro" in resultado.lower() or "zero" in resultado.lower()

    def test_expressao_invalida(self):
        """Testa que expressão inválida retorna mensagem de erro."""
        resultado = calcular.invoke({"expressao": "abc + def"})
        # Deve retornar alguma mensagem de erro (não lançar exceção)
        assert isinstance(resultado, str)

    def test_resultado_inteiro_sem_decimal(self):
        """Testa que resultados inteiros não têm parte decimal desnecessária."""
        resultado = calcular.invoke({"expressao": "4.0 * 2.0"})
        assert resultado == "8"


# ---------------------------------------------------------------------------
# Testes da busca de clima (com mock da API)
# ---------------------------------------------------------------------------


class TestBuscarClima:
    """Testes para a ferramenta `buscar_clima` com mock da API."""

    def test_clima_sem_chave_api(self, monkeypatch):
        """Testa que sem chave de API retorna mensagem informativa."""
        monkeypatch.setattr("agente_ia.ferramentas.OPENWEATHERMAP_API_KEY", "")
        resultado = buscar_clima.invoke({"cidade": "São Paulo"})
        assert "não configurada" in resultado.lower() or "defina" in resultado.lower()

    @patch("agente_ia.ferramentas.requests.get")
    @patch("agente_ia.ferramentas.OPENWEATHERMAP_API_KEY", "chave_fake_123")
    def test_clima_sucesso(self, mock_get):
        """Testa resposta bem-sucedida da API de clima."""
        # Configura o mock para retornar uma resposta simulada
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "main": {
                "temp": 22.5,
                "feels_like": 21.0,
                "humidity": 65,
            },
            "weather": [{"description": "céu limpo"}],
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        resultado = buscar_clima.invoke({"cidade": "São Paulo"})

        assert "22.5" in resultado
        assert "São Paulo" in resultado
        assert "céu limpo" in resultado.lower()

    @patch("agente_ia.ferramentas.requests.get")
    @patch("agente_ia.ferramentas.OPENWEATHERMAP_API_KEY", "chave_fake_123")
    def test_cidade_nao_encontrada(self, mock_get):
        """Testa erro 404 (cidade não encontrada)."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response

        resultado = buscar_clima.invoke({"cidade": "CidadeInexistente123"})

        assert "não encontrada" in resultado.lower() or "erro" in resultado.lower()

    @patch("agente_ia.ferramentas.requests.get")
    @patch("agente_ia.ferramentas.OPENWEATHERMAP_API_KEY", "chave_fake_123")
    def test_erro_de_conexao(self, mock_get):
        """Testa que erro de conexão retorna mensagem de erro."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Sem conexão")

        resultado = buscar_clima.invoke({"cidade": "São Paulo"})

        assert "erro" in resultado.lower() or "conexão" in resultado.lower()


# ---------------------------------------------------------------------------
# Testes da busca na Wikipedia (com mock da API)
# ---------------------------------------------------------------------------


class TestBuscarWikipedia:
    """Testes para a ferramenta `buscar_wikipedia` com mock da API."""

    @patch("agente_ia.ferramentas.requests.get")
    def test_busca_sucesso(self, mock_get):
        """Testa busca bem-sucedida na Wikipedia."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "title": "Python (linguagem de programação)",
            "extract": (
                "Python é uma linguagem de programação de alto nível, "
                "interpretada, de script, imperativa, orientada a objetos, "
                "funcional, de tipagem dinâmica e forte."
            ),
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        resultado = buscar_wikipedia.invoke({"topico": "Python"})

        assert "Python" in resultado
        assert "linguagem" in resultado.lower()

    @patch("agente_ia.ferramentas.requests.get")
    def test_topico_nao_encontrado(self, mock_get):
        """Testa erro 404 (tópico não encontrado)."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response

        resultado = buscar_wikipedia.invoke({"topico": "TopicoInexistente12345"})

        assert "não encontrado" in resultado.lower() or "erro" in resultado.lower()

    @patch("agente_ia.ferramentas.requests.get")
    def test_resumo_longo_truncado(self, mock_get):
        """Testa que resumos muito longos são truncados em 500 caracteres."""
        resumo_longo = "A" * 600  # 600 caracteres
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "title": "Teste",
            "extract": resumo_longo,
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        resultado = buscar_wikipedia.invoke({"topico": "Teste"})

        # O resultado deve ter o título + no máximo 500 chars + "..."
        assert resultado.endswith("...")
        # Verifica que o conteúdo foi truncado
        assert len(resultado) < 600 + len("**Teste**\n")

    @patch("agente_ia.ferramentas.requests.get")
    def test_erro_de_conexao(self, mock_get):
        """Testa que erro de rede retorna mensagem de erro."""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        resultado = buscar_wikipedia.invoke({"topico": "Python"})

        assert "erro" in resultado.lower() or "conexão" in resultado.lower()
