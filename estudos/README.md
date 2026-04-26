# Guia Completo de Estudos - Agentes de IA

## 1. O que é um agente de IA

Um agente de IA é um sistema que:

1. Recebe uma pergunta
2. Decide o que fazer (reasoning)
3. Usa ferramentas (APIs, banco, etc)
4. Retorna uma resposta

Fluxo:

Pergunta → Decisão → Ação → Resposta

---

## 2. ReAct (Reason + Act)

ReAct significa:

- Reason: o modelo pensa
- Act: executa uma ação

Exemplo:

Usuário: "Qual o clima em Salvador?"

Reason: "Preciso buscar o clima"
Act: chamar API de clima

---

## 3. Prompt Engineering

Estrutura de prompt:

- Papel: "Você é um agente de IA"
- Tarefa: "Decida qual ferramenta usar"
- Formato: "Responda com AÇÃO"

---

## 4. APIs

Consumo básico:

- GET → buscar dados
- POST → enviar dados

---

## 5. RAG (Retrieval-Augmented Generation)

Fluxo:

1. Carrega documentos
2. Busca partes relevantes
3. Envia para o modelo
4. Gera resposta baseada nisso

---

## 6. Como explicar na entrevista

"Um agente de IA utiliza LLMs para decidir ações e integrar ferramentas externas. Ele segue um fluxo de reasoning (ReAct), podendo consultar APIs ou bases internas via RAG antes de gerar a resposta final."

---

## 7. Pontos que você deve falar

- diferença entre chatbot e agente
- uso de ferramentas
- limitação do LLM (não acessa internet sozinho)
- importância do contexto (RAG)

---

## 8. Erros comuns

- não separar decisão de execução
- não tratar erro de API
- prompts mal definidos

---

## 9. Evolução futura

- LangChain
- LlamaIndex
- Banco vetorial (FAISS)
- Deploy em cloud
