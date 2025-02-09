from langchain_ollama.llms import OllamaLLM

def invoke_llm(model_name, context, question):
    llm = OllamaLLM(model=model_name)
    response = llm.invoke(f"""
    Você é um chatbot especializado em responder sobre editais de Processo Seletivo da UFPA. 
    Utilizando como base o seguinte contexto: {context}, Responda a seguinte Pergunta: {question} """)
    return response

def generate_response_with_models(models, context, question):
    """
    Gera resumos usando múltiplos modelos de LLM.
    """
    response = [invoke_llm(model_name, context, question) for model_name in models]
    print (f"Respostas dos modelos: {response}\n")
    return response

def evaluate_responses (responses, question):
    """
    Avalia os resumos gerados pelos modelos e seleciona o melhor.
    """
    evaluation_model = OllamaLLM(model="llama3.2:1b")  # Exemplo de modelo central para avaliação
    evaluation_prompt = f""" Você é um avaliador especializado avaliar respostas para perguntas sobre editais de Processo Seletivo da UFPA. Aqui esta uma pergunta {question}, avalie a melhor resposta entre as respostas geradas pelos modelos: {', '.join([f'Response {i + 1}: {response}' for i, response in enumerate(responses)])}

    Escolha a melhor resposta e responda.
    """
    best_response = evaluation_model.invoke(evaluation_prompt)
    return best_response

