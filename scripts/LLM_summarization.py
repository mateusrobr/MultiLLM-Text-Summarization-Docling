from langchain_ollama.llms import OllamaLLM

def invoke_llm(model_name, context, question):
    llm = OllamaLLM(model=model_name)
    response = llm.invoke(f"""
    Você é um chatbot especializado em responder sobre editais de Processo Seletivo da UFPA. 
    Responda com base no seguinte contexto: {context}. 
    Pergunta: {question}
    """)
    return response

def generate_summary_with_models(models, context, question):
    """
    Gera resumos usando múltiplos modelos de LLM.
    """
    summaries = [invoke_llm(model_name, context, question) for model_name in models]
    return summaries

def evaluate_summaries(models, context, summaries, question):
    """
    Avalia os resumos gerados pelos modelos e seleciona o melhor.
    """
    evaluation_model = OllamaLLM(model="llama3.2:3b")  # Exemplo de modelo central para avaliação
    evaluation_prompt = f"""
    Você é um avaliador especializado em resumos. Aqui estão os resumos gerados por diferentes modelos:
    {', '.join([f'Summary {i + 1}: {summary}' for i, summary in enumerate(summaries)])}
    Agora voce vai avaliar o melhor resumo para a seguninte pergunta: {question}
    """
    best_summary = evaluation_model.invoke(evaluation_prompt)
    return best_summary

