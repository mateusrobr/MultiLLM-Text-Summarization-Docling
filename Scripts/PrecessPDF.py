import os
from dotenv import load_dotenv, find_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared, operations
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from typing import Any
from pydantic import BaseModel

load_dotenv(find_dotenv())
unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
unstructured_api_url = os.getenv("UNSTRUCTURED_API_URL")

client = UnstructuredClient(
    api_key_auth=unstructured_api_key, 
    server_url=unstructured_api_url
)

class Element(BaseModel):
    type: str
    page_content: Any

pdf_path = "Editais\Edital_05_2024.pdf"

def pdf_to_text(pdf_path):
    with open(pdf_path, "rb") as f:
        files = shared.Files(
            content=f.read(),
            file_name=pdf_path
        )

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=files,
            strategy=shared.Strategy.HI_RES,
            hi_res_model_name="yolox",
            skip_infer_table_types=[],
            pdf_infer_table_structure=True
        ),
        unstructured_api_key=unstructured_api_key
    )

    try:
        resp = client.general.partition(request=req)
        elements = dict_to_elements(resp.elements)
    except SDKError as e:
        print(e)

    categorized_elements = []

    for element in elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", page_content=str(element.metadata.text_as_html)))
        elif "unstructured.documents.elements.NarrativeText" in str(type(element)):
            categorized_elements.append(Element(type="text", page_content=str(element)))
        elif "unstructured.documents.elements.ListItem" in str(type(element)):
            categorized_elements.append(Element(type="text", page_content=str(element)))
        elif "unstructured.documents.elements.Title" in str(type(element)):
            categorized_elements.append(Element(type="text", page_content=str(element)))
        elif "unstructured.documents.elements.Address" in str(type(element)):
            categorized_elements.append(Element(type="text", page_content=str(element)))
        elif "unstructured.documents.elements.EmailAddress" in str(type(element)):
            categorized_elements.append(Element(type="text", page_content=str(element)))
        elif "unstructured.documents.elements.Header" in str(type(element)):
            categorized_elements.append(Element(type="CodeSnippet", page_content=str(element)))
        elif "unstructured.documents.elements.CodeSnippet" in str(type(element)):
            categorized_elements.append(Element(type="text", page_content=str(element)))
        elif "unstructured.documents.elements.UncategorizedText" in str(type(element)):
            categorized_elements.append(Element(type="text", page_content=str(element)))
        
        # Se unstructured.documents.elements.image = PrecessaImagem

    return categorized_elements

def save_elements_to_txt(elements, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for element in elements:
            f.write(f"Type: {element.type}\n")
            f.write(f"Content: {element.page_content}\n")
            f.write("\n")

if __name__ == "__main__":
    pdf_path = "Editais\Edital_05_2024.pdf"
    output_path = "output.txt"
    elements = pdf_to_text(pdf_path)
    save_elements_to_txt(elements, output_path)
    print(f"Elements saved to {output_path}")
