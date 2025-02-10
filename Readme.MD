# MultiLLM-Text-Summarization

Este repositório contém um aplicativo web desenvolvido com Flask que permite aos usuários interagir com um chatbot e fazer upload de arquivos PDF para análise. O sistema utiliza um banco de dados vetorial para armazenar e recuperar documentos, e gera respostas usando múltiplos modelos de linguagem.


## Requisitos

- Python 3.x
- Flask
- Bibliotecas adicionais listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
   ```sh
   git clone <URL_DO_REPOSITORIO>
   cd MultiLLM-Text-Summarization
   ```

2. Crie um ambiente virtual e ative-o:
   ```sh
   python -m venv venv
   source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
   ```

3. Instale as dependências:
   ```sh
   pip install -r requirements.txt
   ```

## Uso

1. Inicie o servidor Flask:
   ```sh
   python app.py
   ```

2. Abra um navegador web e acesse `http://127.0.0.1:5000/`.

3. Use a interface web para interagir com o chatbot e fazer upload de arquivos PDF.

## Estrutura dos Arquivos

### `app.py`

Este é o arquivo principal do aplicativo Flask. Ele define as rotas e a lógica para lidar com mensagens do chatbot e uploads de PDF.

### `ChatBot.py`

Contém a lógica do chatbot, incluindo a função `chatbot` que processa as mensagens dos usuários e gera respostas usando múltiplos modelos de linguagem.

### `DataBase.py`

Contém funções para inicializar o banco de dados vetorial, carregar documentos PDF e armazená-los no banco de dados.

### `templates/index.html`

O arquivo HTML que define a interface do usuário para o chatbot e o upload de PDF.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.
