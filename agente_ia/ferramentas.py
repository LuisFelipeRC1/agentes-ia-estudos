"""
ferramentas.py - Ferramentas externas para o agente
=====================================================
Ferramentas (tools) são funções que o agente pode invocar durante seu
raciocínio para buscar informações ou executar ações no mundo real.

O decorador @tool do LangChain:
- Transforma uma função Python comum em uma ferramenta compatível com agentes.
- O docstring da função vira a descrição que o LLM usa para decidir quando
  chamar a ferramenta — escreva docstrings claros e descritivos!
- Os parâmetros da função se tornam os parâmetros da ferramenta.

Ferramentas implementadas:
1. buscar_clima       - Consulta temperatura e clima atual via API REST
2. calcular           - Avalia expressões matemáticas de forma segura
3. buscar_wikipedia   - Busca resumo de um tópico na Wikipedia (API pública)
"""

import ast
import operator
import re
from typing import Union

import requests
from langchain_core.tools import tool

from agente_ia.config import OPENWEATHERMAP_API_KEY


# ---------------------------------------------------------------------------
# Ferramenta 1: Clima atual
# ---------------------------------------------------------------------------

@tool
def buscar_clima(cidade: str) -> str:
    """
    Retorna a temperatura atual e a descrição do clima para uma cidade.
    Use esta ferramenta quando o usuário perguntar sobre o clima ou temperatura
    de uma cidade específica.

    Args:
        cidade: Nome da cidade (ex: "São Paulo", "Rio de Janeiro", "Lisboa").

    Returns:
        String com temperatura em °C e descrição do clima, ou mensagem de erro.
    """
    if not OPENWEATHERMAP_API_KEY:
        return (
            "Ferramenta de clima não configurada. "
            "Defina OPENWEATHERMAP_API_KEY no arquivo .env."
        )

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": cidade,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric",   # temperatura em Celsius
        "lang": "pt_br",     # descrição em português
    }

    try:
        resposta = requests.get(url, params=params, timeout=10)
        resposta.raise_for_status()
        dados = resposta.json()

        temperatura = dados["main"]["temp"]
        sensacao = dados["main"]["feels_like"]
        descricao = dados["weather"][0]["description"]
        umidade = dados["main"]["humidity"]

        return (
            f"Clima em {cidade}: {descricao.capitalize()}\n"
            f"Temperatura: {temperatura:.1f}°C (sensação: {sensacao:.1f}°C)\n"
            f"Umidade: {umidade}%"
        )
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return f"Cidade '{cidade}' não encontrada. Verifique o nome e tente novamente."
        return f"Erro ao consultar a API de clima: {e}"
    except requests.exceptions.RequestException as e:
        return f"Erro de conexão ao consultar o clima: {e}"


# ---------------------------------------------------------------------------
# Ferramenta 2: Calculadora segura
# ---------------------------------------------------------------------------

# Operadores matemáticos permitidos — limita o que pode ser avaliado
_OPERADORES_PERMITIDOS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def _avaliar_expressao(node: ast.AST) -> Union[int, float]:
    """Avalia recursivamente um nó AST de expressão aritmética."""
    if isinstance(node, ast.Constant):
        # Número literal (ex: 42, 3.14)
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Tipo não suportado: {type(node.value)}")

    if isinstance(node, ast.BinOp):
        # Operação binária (ex: a + b)
        op_type = type(node.op)
        if op_type not in _OPERADORES_PERMITIDOS:
            raise ValueError(f"Operador não permitido: {op_type.__name__}")
        esquerda = _avaliar_expressao(node.left)
        direita = _avaliar_expressao(node.right)
        return _OPERADORES_PERMITIDOS[op_type](esquerda, direita)

    if isinstance(node, ast.UnaryOp):
        # Operador unário (ex: -5)
        op_type = type(node.op)
        if op_type not in _OPERADORES_PERMITIDOS:
            raise ValueError(f"Operador não permitido: {op_type.__name__}")
        return _OPERADORES_PERMITIDOS[op_type](_avaliar_expressao(node.operand))

    raise ValueError(f"Expressão não suportada: {type(node).__name__}")


@tool
def calcular(expressao: str) -> str:
    """
    Calcula o resultado de uma expressão matemática.
    Use esta ferramenta para realizar cálculos numéricos precisos.
    Suporta: adição (+), subtração (-), multiplicação (*), divisão (/),
    potenciação (**) e módulo (%).

    Args:
        expressao: Expressão matemática como string (ex: "2 + 3 * 4", "10 / 2").

    Returns:
        Resultado do cálculo ou mensagem de erro.
    """
    # Remove espaços e caracteres potencialmente perigosos antes de parsear
    expressao_limpa = re.sub(r"[^0-9+\-*/().\s%*]", "", expressao)

    try:
        arvore = ast.parse(expressao_limpa, mode="eval")
        resultado = _avaliar_expressao(arvore.body)

        # Formata o resultado: inteiro se não tiver parte fracionária
        if isinstance(resultado, float) and resultado.is_integer():
            return str(int(resultado))
        return str(resultado)

    except ZeroDivisionError:
        return "Erro: divisão por zero não é permitida."
    except (ValueError, TypeError, SyntaxError) as e:
        return f"Erro ao calcular '{expressao}': {e}"


# ---------------------------------------------------------------------------
# Ferramenta 3: Busca na Wikipedia
# ---------------------------------------------------------------------------

@tool
def buscar_wikipedia(topico: str) -> str:
    """
    Busca um resumo sobre um tópico na Wikipedia em português.
    Use esta ferramenta quando precisar de informações factuais ou definições
    sobre pessoas, lugares, conceitos, eventos históricos, etc.

    Args:
        topico: O tópico a pesquisar (ex: "Inteligência Artificial", "Python").

    Returns:
        Resumo do artigo da Wikipedia ou mensagem de erro.
    """
    url = "https://pt.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(
        topico
    )

    try:
        resposta = requests.get(url, timeout=10)
        resposta.raise_for_status()
        dados = resposta.json()

        titulo = dados.get("title", topico)
        resumo = dados.get("extract", "Resumo não disponível.")

        # Limita a 500 caracteres para não sobrecarregar o contexto do LLM
        if len(resumo) > 500:
            resumo = resumo[:500] + "..."

        return f"**{titulo}**\n{resumo}"

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return f"Tópico '{topico}' não encontrado na Wikipedia."
        return f"Erro ao acessar a Wikipedia: {e}"
    except requests.exceptions.RequestException as e:
        return f"Erro de conexão com a Wikipedia: {e}"


# ---------------------------------------------------------------------------
# Lista de todas as ferramentas disponíveis para o agente
# ---------------------------------------------------------------------------

FERRAMENTAS_DISPONIVEIS = [
    buscar_clima,
    calcular,
    buscar_wikipedia,
]
