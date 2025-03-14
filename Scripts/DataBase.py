from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import OllamaEmbeddings
#from langchain_community.vectorstores import Chroma
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from Scripts.PrecessPDF import extract_images, get_images_description, pdf_to_doc
from chromadb import PersistentClient

default_persist_directory = "./chroma_langchain_db"

default_persist_multimodal_directory = "./chroma_multimodal_db"

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
            pdf_text = pdf_to_doc(pdf_path)
            print(f"Carregando documento {pdf_path}...")
            splitted_text = text_splitter.split_documents(pdf_text)
            all_splitts.extend(splitted_text)
            print(f"Documento {pdf_path} carregado com sucesso!")
            #extract_images(pdf_path)
            
        except Exception as e:
            print(f"Erro ao carregar o documento {pdf_path}: {e}")
    return all_splitts


def storageInChroma(texts, persist_directory=default_persist_directory, vector_db=None):
    if vector_db is None:
        vector_db = initializeChromaDB(persist_directory)
        if vector_db is None:
            return None

    try:
        vector_db.add_documents(texts)
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
        print(paths)
        splitted_texts = loadDocuments(paths)
        for path in paths:

            image_ids_paths = extract_images(path)
            ids_res_dict = get_images_description(image_ids_paths)

        
        if not splitted_texts:
            print("Falha ao carregar os documentos. Verifique os caminhos e tente novamente.")
            return None
        
        print("Armazenando documentos no ChromaDB...")
        vectorstore = storageInChroma(splitted_texts, persist_directory=persist_directory)
        multimodal_store = loadAndStoreImages(image_ids_paths,ids_res_dict)
        
        if vectorstore and multimodal_store:
            print(f"Documentos armazenados com sucesso no ChromaDB! Dados persistidos em: {persist_directory}")
        else:
            print("Falha ao armazenar os documentos no ChromaDB.")
        
        return vectorstore
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None
    

def initializeChromaMultimodal(persist_directory= default_persist_multimodal_directory):
    '''
    Inicializa um DB multimodal

    Args:
        Caminho para o banco de dados ser guardado na memória

    Returns:
        Banco de dados inicializado
    '''

    try:
        chroma_client = PersistentClient(path=default_persist_multimodal_directory)

        image_loader = ImageLoader()
        multimodal_emb_func = OpenCLIPEmbeddingFunction()

        multimodal_db = chroma_client.get_or_create_collection(name='multimodal_db',embedding_function=multimodal_emb_func, data_loader=image_loader)
    
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None

    return multimodal_db

'''def loadAndStoreImages(images_ids_and_paths):
    
    Método para guardar imagens no banco de dados multimodal

    Args:
        Lista que cada unidade da lista contém uma outra lista que possui o ID e o path da imagem

    Returns:
        Retorna o banco de dados atualizado caso dê certo e retorna none caso nao de certo


    multimodal_db = initializeChromaMultimodal(default_persist_directory)

    try: 
        for id in images_ids_and_paths:
            multimodal_db.add(
                ids=[id[0]],
                uris=[id[1]]
            )
    except Exception as e:
        print(f"Ocorreu um erro {e}")
    return multimodal_db'''



def loadAndStoreImages(images_ids_and_paths,ids_res_dict):
    '''
    Método para guardar imagens no banco de dados multimodal

    Args:
        Lista que cada unidade da lista contém uma outra lista que possui o ID e o path da imagem

    Returns:
        Retorna o banco de dados atualizado caso dê certo e retorna none caso nao de certo
    '''

    multimodal_db = initializeChromaMultimodal(default_persist_directory)

    try: 
        for id in images_ids_and_paths:
            multimodal_db.add(
                ids=[id[0]],
                uris=[id[1]],
                metadatas={
                    "description": ids_res_dict[id[0]]['message']['content']
                }
            )
    except Exception as e:
        print(f"Ocorreu um erro {e}")
    return multimodal_db

