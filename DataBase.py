from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

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
            embedding_function=OllamaEmbeddings(model="all-minilm:33m"),
            persist_directory=persist_directory,  # Where to save data locally, remove if not necessary
        )
        print(f"Banco de dados vetorial inicializado com sucesso!")
        return vector_store
    except Exception as e:
        print(f"Erro ao inicializar o banco de dados vetorial: {e}")
        return None


def loadDocument(pdf_path):
    """
    Carrega um documento PDF e o divide em partes menores para processamento.
    
    Args:
        pdf_path (str): Caminho para o arquivo PDF.
    
    Returns:
        list: Uma lista de documentos divididos em partes.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        doc = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2200,
            chunk_overlap=100,
            add_start_index=True
        )
        all_splitts = text_splitter.split_documents(doc)
        return all_splitts
    except Exception as e:
        print(f"Erro ao carregar o documento: {e}")
        return None


def storageInChroma(documents, persist_directory=default_persist_directory, vector_db=initializeChromaDB()):
    """
    Armazena os documentos em um banco de dados vetorial usando ChromaDB com persistência.
    
    Args:
        documents (list): Lista de documentos processados.
        persist_directory (str): Diretório para persistir os dados do ChromaDB.
        vector_db (Chroma): O banco de dados vetorial para armazenar os documentos.
    
    Returns:
        Chroma: O banco de dados vetorial com os documentos armazenados.
    """
    try:        
        # Configurar o banco de dados Chroma
        vector_db.add_documents(documents=documents)
        
        print(f"Documentos armazenados no diretório: {persist_directory}")
        return vector_db
    except Exception as e:
        print(f"Erro ao armazenar os documentos no ChromaDB: {e}")
        return None


def loadAndStoreDocument(path, persist_directory=default_persist_directory):
    """
    Solicita o caminho de um documento ao usuário, carrega o conteúdo e o armazena no ChromaDB.
    
    Args:
        persist_directory (str): Diretório para persistir os dados do ChromaDB.
    
    Returns:
        Chroma: O banco de dados vetorial com os documentos armazenados.
    """
    try:
        # Carregar o documento
        print("Carregando documento...")
        documents = loadDocument(path)
        
        if not documents:
            print("Falha ao carregar o documento. Verifique o caminho e tente novamente.")
            return None
        
        # Armazenar no ChromaDB
        print("Armazenando documentos no ChromaDB...")
        vectorstore = storageInChroma(
            documents, 
            persist_directory=persist_directory
        )
        
        if vectorstore:
            print(f"Documento armazenado com sucesso no ChromaDB! Dados persistidos em: {persist_directory}")
        else:
            print("Falha ao armazenar os documentos no ChromaDB.")
        
        return vectorstore
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None
