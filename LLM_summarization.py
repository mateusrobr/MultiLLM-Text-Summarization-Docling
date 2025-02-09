from langchain_ollama.llms import OllamaLLM

def invoke_llm(model_name, context, question):
    llm = OllamaLLM(model=model_name)
    response = llm.invoke(f"""
    Você é um chatbot especializado em responder sobre editais de Processo Seletivo da UFPA. 
    Utilizando como base o seguinte contexto: {context}, Responda a seguinte Pergunta: {question}. Responda utilizando a lingua portuguesa""")
    response = (f'{model_name}: {response}')
    return response

def generate_response_with_models(models, context, question):
    """
    Gera Respostas usando múltiplos modelos de LLM.
    """
    response = [invoke_llm(model_name, context, question) for model_name in models]
    print (f"\nRespostas dos modelos: {response}\n")
    return response

def evaluate_responses (responses, question):
    """
    Avalia as respostas geradas pelos modelos e seleciona o melhor.
    """
    evaluation_model = OllamaLLM(model="llama3.2:1b")  # Exemplo de modelo central para avaliação
    evaluation_prompt = f"""Você é um chatbot e esta conversado com o usuario, responda cordialmente e em portugues, além disso Você é um avaliador especializado avaliar respostas para perguntas sobre editais de Processo Seletivo da UFPA. Aqui esta uma pergunta {question}, avalie qual é a melhor resposta entre as respostas geradas pelos modelos: {', '.join([f'Response {i + 1}: {response}' for i, response in enumerate(responses)])}. Escolha uma das respostas como a melhor resposta e mostre. Não invente informações, não acrescente informações, apenas escolha a melhor resposta dentre as apresentadas e mostre-a para mim."""
    best_response = evaluation_model.invoke(evaluation_prompt)
    return best_response

