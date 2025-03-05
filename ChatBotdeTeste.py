from Scripts.DataBase import initializeChromaDB, initializeChromaMultimodal, loadAndStoreDocuments
from Scripts.LLM_summarization import generate_response_with_models, evaluate_responses
from Scripts.PrecessPDF import show_image

vectorstore = initializeChromaDB()
image_vector_store = initializeChromaMultimodal()

def main_menu():
    print("\n=========== Menu do ChatBot ===========")
    print("1 - Enviar um documento PDF para análise")
    print("2 - Conversar com o ChatBot")
    print("3 - Sair")
    print("=======================================\n")

def chatbot():
    print("\n============ Iniciando ChatBot =============")
    print("ChatBot -> Olá, sou um ChatBot e estou aqui para te ajudar com dúvidas sobre editais da UFPA.")
    print("Digite 'exit' a qualquer momento para voltar ao menu principal.\n")
    
    
    models = ["qwen:4b","falcon:7b"]

    message = ''
    while message.lower() != "exit":
        message = input("Usuário -> ").strip()
        
        retrive = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )        

        retrive_doc = retrive.invoke(message)
        context = ' '.join([doc.metadata for doc in retrive_doc])

        print(f"----------------------------------------------------------------\nContexto: {retrive_doc['metadata']}\n----------------------------------------------------------------")

        if image_vector_store.count() > 1:
            results = image_vector_store.query(
                query_texts=[message],
                n_results=1,
                include=[ 'distances', 'metadatas', 'data', 'uris']
            )
            if results['distances'][0][0] < 1.5:
                show_image(results[0][0])
    
    # Geração de responstas usando múltiplos modelos
    #summaries = generate_response_with_models(models, context, message)

    # Avaliação e seleção da melhor resposta
    #best_response = evaluate_responses(summaries, message)

    #return best_response

def main():
    while True:
        main_menu()
        choice = input("Escolha uma opção: ").strip()

        if choice == "1":
            pdf_paths = input("\nDigite os caminhos completos dos arquivos PDF separados por vírgula: ").strip().split(',')
            pdf_paths = [path.strip() for path in pdf_paths]
            loadAndStoreDocuments(paths=pdf_paths)

        elif choice == "2":
            if 'vectorstore' not in globals() or vectorstore is None:
                print("\nErro: Nenhum documento foi carregado no banco de dados ainda.")
                print("Por favor, use a opção 1 para carregar um documento antes de conversar com o ChatBot.")
            else:
                chatbot()

        elif choice == "3":
            print("\nEncerrando o programa. Até logo!")
            break

        else:
            print("\nOpção inválida. Por favor, escolha uma opção válida.")

if __name__ == "__main__":
    main()