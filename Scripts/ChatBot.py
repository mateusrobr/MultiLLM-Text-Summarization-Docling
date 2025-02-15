from Scripts.DataBase import initializeChromaDB
from Scripts.LLM_summarization import generate_response_with_models, evaluate_responses

vectorstore = initializeChromaDB()

def main_menu():
    print("\n=========== Menu do ChatBot ===========")
    print("1 - Enviar um documento PDF para análise")
    print("2 - Conversar com o ChatBot")
    print("3 - Sair")
    print("=======================================\n")

def chatbot(message):
    '''print("\n============ Iniciando ChatBot =============")
    print("ChatBot -> Olá, sou um ChatBot e estou aqui para te ajudar com dúvidas sobre editais da UFPA.")
    print("Digite 'exit' a qualquer momento para voltar ao menu principal.\n")'''
    
    models = ["qwen:4b","falcon:7b"]
    

    retrive = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )


    retrive_doc = retrive.invoke(message)
    context = ' '.join([doc.page_content for doc in retrive_doc])
        

    # Geração de responstas usando múltiplos modelos
    summaries = generate_response_with_models(models, context, message)

    # Avaliação e seleção da melhor resposta
    best_response = evaluate_responses(summaries, message)

    return best_response