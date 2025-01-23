from langchain_ollama.llms import OllamaLLM
def generate_summary_with_models(models, context, question):
    """
    Gera resumos usando múltiplos modelos de LLM.
    """
    summaries = []
    for model_name in models:
        llm = OllamaLLM(model=model_name)
        response = llm.invoke(f"""
        Você é um chatbot especializado em responder sobre editais de Processo Seletivo da UFPA. 
        Responda com base no seguinte contexto: {context}. 
        Pergunta: {question}
        """)
        summaries.append(response)
    return summaries

def evaluate_summaries(models, context, summaries, question):
    """
    Avalia os resumos gerados pelos modelos e seleciona o melhor.
    """
    evaluation_model = OllamaLLM(model="llama3.2:3b")  # Exemplo de modelo central para avaliação
    evaluation_prompt = f"""
    Você é um avaliador especializado em resumos. Aqui estão os resumos gerados por diferentes modelos:
    {', '.join([f'Summary {i + 1}: {summary}' for i, summary in enumerate(summaries)])}
    
    Pergunta: {question}
    Escolha o melhor resumo e responda.
    """
    best_summary = evaluation_model.invoke(evaluation_prompt)
    return best_summary

