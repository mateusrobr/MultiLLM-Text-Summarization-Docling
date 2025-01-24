from langchain_ollama.llms import OllamaLLM
from DataBase import loadAndStoreDocument, initializeChromaDB
from LLM_summarization import generate_summary_with_models, evaluate_summaries

vectorstore = initializeChromaDB()

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
    
    # Lista de modelos para geração e avaliação
    # models = ["tinyllama:latest", "falcon:7b", "qwen:4b"]
    models = ["tinyllama:latest", "falcon:7b"]

    while True:
        question = input("Você -> ")
        if question.lower() == "exit":
            print("\nVoltando ao menu principal...")
            break

        retrive = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        retrive_doc = retrive.invoke(question)
        context = ' '.join([doc.page_content for doc in retrive_doc])

        # Geração de resumos usando múltiplos modelos
        summaries = generate_summary_with_models(models, context, question)

        # Avaliação e seleção do melhor resumo
        best_summary = evaluate_summaries(models, context, summaries, question)

        print(f"ChatBot -> {best_summary}")

def main():
    while True:
        main_menu()
        choice = input("Escolha uma opção: ").strip()

        if choice == "1":
            pdf_path = input("\nDigite o caminho completo do arquivo PDF: ").strip()
            loadAndStoreDocument(path=pdf_path)

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

