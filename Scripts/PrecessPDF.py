import os
from dotenv import load_dotenv, find_dotenv
from langchain_unstructured import UnstructuredLoader
from unstructured_client import UnstructuredClient

def pdf_to_doc(file_path):
    """
    Carregar um documento PDF para análise.
    
    Args:
        file_path (str): O caminho do arquivo PDF.
    
    Returns:
        list: Uma lista de documentos carregados.
    """
    
    load_dotenv(find_dotenv())
    unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
    unstructured_api_url = os.getenv("UNSTRUCTURED_API_URL")
    
    
    loader = UnstructuredLoader(
        file_path=file_path,
        strategy="hi_res",
        partition_via_api=True,
        coordinates=True,
        client = UnstructuredClient(
            api_key_auth=unstructured_api_key, 
            server_url=unstructured_api_url)
    )

    docs = []

    for doc in loader.lazy_load():
        docs.append(doc)
    
    return docs

