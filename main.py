def buscar_clima(cidade):
    return f"O clima em {cidade} está ensolarado com 28°C."


def buscar_cep(cep):
    return f"O CEP {cep} corresponde à Praça da Sé, São Paulo."


def agente(pergunta):
    pergunta_lower = pergunta.lower()

    # Etapa 1 - decisão (simulação de LLM)
    if "clima" in pergunta_lower:
        acao = "buscar_clima"
    elif "cep" in pergunta_lower:
        acao = "buscar_cep"
    else:
        acao = "llm"

    print("[DEBUG] Decisão do agente:", acao)

    # Etapa 2 - execução da ferramenta
    if acao == "buscar_clima":
        resultado = buscar_clima("Salvador")
    elif acao == "buscar_cep":
        resultado = buscar_cep("01001000")
    else:
        resultado = "Resposta direta sem ferramenta"

    print("[DEBUG] Resultado da ferramenta:", resultado)

    # Etapa 3 - resposta final
    resposta_final = f"Resposta final: {resultado}"

    return resposta_final


if __name__ == "__main__":
    pergunta = "Qual o clima em Salvador?"
    print(agente(pergunta))
