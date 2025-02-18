import os
from dotenv import load_dotenv, find_dotenv
import fitz
from langchain_unstructured import UnstructuredLoader
from pdf2image import convert_from_path
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

def extract_images(pdf_path):
    """
    Extrai imagens de um PDF
    
    Args:
        pdf_path (str): O caminho do arquivo PDF.
    
    """

    doc = fitz.open(pdf_path)

    output_folder = "extracted_images"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    documents_number = len(os.listdir(output_folder))
    image_paths = []
    for page_number in range(len(doc)):  # Itera sobre todas as páginas do PDF
        page = doc[page_number]
        images = page.get_images(full=True)  # Obtém todas as imagens embutidas na página
            
        for img in images:
            xref = img[0]  # ID da imagem no PDF
            base_image = doc.extract_image(xref)  # Extrai a imagem
            image_bytes = base_image["image"]  # Obtém os bytes da imagem
            image_ext = base_image["ext"]  # Obtém a extensão da imagem (png, jpeg, etc.)

            # Criar o nome do arquivo
            image_filename = f"image_{documents_number + 1}.{image_ext}"
            documents_number += 1
            image_path = os.path.join(output_folder, image_filename)
            image_paths.append(image_path)
            # Salvar a imagem extraída
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

    return image_paths

def get_images_description(images_path):
    descriptions = []
    for path in image_paths:

        res = ollama.chat(
            model='llava',
            messages=[
                {
                    'role': 'user',
                    'content': 'Describe this image',
                    'images': [f'{path}']
                }
            ]
        )

        descriptions.append(res)
    
    return description