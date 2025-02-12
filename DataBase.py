from xml.dom.minidom import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import pdfplumber
from langchain_community.document_loaders import PDFPlumberLoader
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


def extract_only_text(pdf_path):

    extracted_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:

            extracted_text.append(page.extract_text())
    
    return "\n".join(extracted_text)




def loadDocuments(pdf_paths):
    """
    Carrega múltiplos documentos PDF e os divide em partes menores para processamento.
    
    Args:
        pdf_paths (list): Lista de caminhos para os arquivos PDF.
    
    Returns:
        list: Uma lista de documentos divididos em partes.
    """
    all_splitts = []
    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                add_start_index=True
            )
    for pdf_path in pdf_paths:
        try:
            
            pdf_text = extract_only_text(pdf_path)
            splitted_text = text_splitter.split_text(pdf_text)
            

            '''text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                add_start_index=True
            )'''

            all_splitts.extend(splitted_text)
        except Exception as e:
            print(f"Erro ao carregar o documento {pdf_path}: {e}")
    return all_splitts

'''def storageInChroma(documents, persist_directory=default_persist_directory, vector_db=None):
    """
    Armazena os documentos em um banco de dados vetorial usando ChromaDB com persistência.
    
    Args:
        documents (list): Lista de documentos processados.
        persist_directory (str): Diretório para persistir os dados do ChromaDB.
        vector_db (Chroma): O banco de dados vetorial para armazenar os documentos.
    
    Returns:
        Chroma: O banco de dados vetorial com os documentos armazenados.
    """
    if vector_db is None:
        vector_db = initializeChromaDB(persist_directory)
        if vector_db is None:
            return None

    try:
        vector_db.add_documents(documents=documents)
        print(f"Documentos armazenados no diretório: {persist_directory}")
        return vector_db
    except Exception as e:
        print(f"Erro ao armazenar os documentos no ChromaDB: {e}")
        return None'''

def storageInChroma(texts, persist_directory=default_persist_directory, vector_db=None):
    """
    Armazena os documentos em um banco de dados vetorial usando ChromaDB com persistência.
    
    Args:
        documents (list): Lista de documentos processados.
        persist_directory (str): Diretório para persistir os dados do ChromaDB.
        vector_db (Chroma): O banco de dados vetorial para armazenar os documentos.
    
    Returns:
        Chroma: O banco de dados vetorial com os documentos armazenados.
    """
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
    Solicita os caminhos de documentos ao usuário, carrega o conteúdo e os armazena no ChromaDB.
    
    Args:
        paths (list): Lista de caminhos para os arquivos PDF.
        persist_directory (str): Diretório para persistir os dados do ChromaDB.
    
    Returns:
        Chroma: O banco de dados vetorial com os documentos armazenados.
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

'''def loadAndStoreDocuments(paths, persist_directory=default_persist_directory):
    """
    Solicita os caminhos de documentos ao usuário, carrega o conteúdo e os armazena no ChromaDB.
    
    Args:
        paths (list): Lista de caminhos para os arquivos PDF.
        persist_directory (str): Diretório para persistir os dados do ChromaDB.
    
    Returns:
        Chroma: O banco de dados vetorial com os documentos armazenados.
    """
    try:
        print("Carregando documentos...")
        documents = loadDocuments(paths)
        
        if not documents:
            print("Falha ao carregar os documentos. Verifique os caminhos e tente novamente.")
            return None
        
        print("Armazenando documentos no ChromaDB...")
        vectorstore = storageInChroma(documents, persist_directory=persist_directory)
        
        if vectorstore:
            print(f"Documentos armazenados com sucesso no ChromaDB! Dados persistidos em: {persist_directory}")
        else:
            print("Falha ao armazenar os documentos no ChromaDB.")
        
        return vectorstore
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None'''
