from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from Scripts.PrecessPDF import pdf_to_text

default_persist_directory = "./chroma_langchain_db"

def initializeChromaDB(persist_directory=default_persist_directory):
    """
    Inicializa um banco de dados vetorial com persistência usando ChromaDB.
    
    Args:
        persist_directory (str): Diretório para persistir os dados do ChromaDB.
    
    Returns:
        Chroma: O banco de dados vetorial inicializado.
    """
    try:
        vector_store = Chroma(
            collection_name="Editais_UFPA",
            embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"),
            persist_directory=persist_directory,
        )
        print(f"Banco de dados vetorial inicializado com sucesso!")
        return vector_store
    except Exception as e:
        print(f"Erro ao inicializar o banco de dados vetorial: {e}")
        return None


def loadDocuments(pdf_paths):
    all_splitts = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True
    )
    
    for pdf_path in pdf_paths:
        try:
            pdf_text = pdf_to_text(pdf_path)
            splitted_text = text_splitter.split_text(pdf_text)
            all_splitts.extend(splitted_text)
            print(f"Documento {pdf_path} carregado com sucesso!")
            
        except Exception as e:
            print(f"Erro ao carregar o documento {pdf_path}: {e}")
    return all_splitts


def storageInChroma(texts, persist_directory=default_persist_directory, vector_db=None):
    if vector_db is None:
        vector_db = initializeChromaDB(persist_directory)
        if vector_db is None:
            return None

    try:
        vector_db.add_texts(texts)
        print(f"Documentos armazenados no diretório: {persist_directory}")
        return vector_db
    except Exception as e:
        print(f"Erro ao armazenar os documentos no ChromaDB: {e}")
        return None

def loadAndStoreDocuments(paths, persist_directory=default_persist_directory):
    """
    Carrega documentos a partir dos caminhos fornecidos, divide os textos e armazena-os no ChromaDB.
    Args:
        paths (list): Lista de caminhos para os documentos a serem carregados.
        persist_directory (str, optional): Diretório onde os dados serão persistidos. 
                                           O valor padrão é `default_persist_directory`.
    Returns:
        object: Retorna o objeto `vectorstore` se os documentos forem armazenados com sucesso no ChromaDB, 
                caso contrário, retorna None.
    Raises:
        Exception: Captura e imprime qualquer exceção que ocorra durante o processo de carregamento e armazenamento.
    """
    try:
        print("Carregando documentos...")
        splitted_texts = loadDocuments(paths)
        
        if not splitted_texts:
            print("Falha ao carregar os documentos. Verifique os caminhos e tente novamente.")
            return None
        
        print("Armazenando documentos no ChromaDB...")
        vectorstore = storageInChroma(splitted_texts, persist_directory=persist_directory)
        
        if vectorstore:
            print(f"Documentos armazenados com sucesso no ChromaDB! Dados persistidos em: {persist_directory}")
        else:
            print("Falha ao armazenar os documentos no ChromaDB.")
        
        return vectorstore
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None