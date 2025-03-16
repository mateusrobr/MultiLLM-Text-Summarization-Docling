import os
from dotenv import load_dotenv, find_dotenv
import fitz
from langchain_unstructured import UnstructuredLoader
from unstructured_client import UnstructuredClient
import ollama
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import  InputFormat
from docling_core.types.doc import PictureItem

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
        client=UnstructuredClient(
            api_key_auth=unstructured_api_key, 
            server_url=unstructured_api_url
        )
    )
    
    docs = loader.load_and_split()  # Carrega e divide o documento PDF em páginas
    return docs


#def extract_images(pdf_path):
    """
    Extrai imagens de um PDF
    
    Args:
        pdf_path (str): O caminho do arquivo PDF.
    
    Returns:
        list: Uma lista de caminhos das imagens extraídas.
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
            documents_number += 1
            image_filename = f"image_{documents_number}.{image_ext}"
            image_id = str(documents_number)
            image_path = os.path.join(output_folder, image_filename)
            image_paths.append([image_id, image_path])

            # Salvar a imagem extraída
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

    return image_paths

def extract_images(pdf_path):
    """
    Extrai imagens de um PDF
    
    Args:
        pdf_path (str): O caminho do arquivo PDF.
    
    Returns:
        list: Uma lista de caminhos das imagens extraídas.
    """
    IMAGE_RESOLUTION_SCALE = 2.0

    input_doc_path = Path(pdf_path)
    output_dir = Path(r".\exemplo_docling")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True


    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    conv_res = doc_converter.convert(input_doc_path)

    documents_number = len(os.listdir(output_dir))
    image_paths = []
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, PictureItem):
            documents_number += 1
            element_image_filename = (
                output_dir / f"picture-{documents_number}.png"
            )
            image_id = str(documents_number)
            image_paths.append([image_id, element_image_filename])
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

    return image_paths


def get_images_description(images_path_and_id):
    """
    Obtém descrições para uma lista de imagens.
    
    Args:
        images_path_and_id (list): Lista de caminhos e IDs das imagens.
    
    Returns:
        dict: Dicionário com IDs de imagens como chaves e descrições como valores.
    """
    descriptions = {}
    for path_and_id in images_path_and_id:
        res = ollama.chat(
            model='llava:latest',
            messages=[
                {
                    'role': 'user',
                    'content': 'Descreva esta imagem',
                    'images': [f'{path_and_id[1]}']
                }
            ]
        )
        descriptions[path_and_id[0]] = res['message']['content']
    
    #print(f'------{descriptions}--------')
    
    return descriptions


def show_image(path):
    """
    Exibe uma imagem.
    
    Args:
        path (str): O caminho da imagem.
    """
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.axis('off')  
    plt.show()