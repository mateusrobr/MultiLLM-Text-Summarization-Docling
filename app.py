from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Configurações de upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Armazena mensagens do chat
chat_history = []

def allowed_file(filename):
    """Verifica se o arquivo tem a extensão permitida (PDF)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['POST'])
def send_message():
    data = request.get_json()
    message = data.get('message')
    
    if message:
        chat_entry = {'message': message}
        chat_entry_bot = {'message': "teste"}
        chat_history.append(chat_entry)
        chat_history.append(chat_entry_bot)
        return jsonify({'status': 'success', 'chat': chat_history})
    
    return jsonify({'status': 'error', 'message': 'Invalid input'}), 400

@app.route('/get_messages', methods=['GET'])
def get_messages():
    return jsonify(chat_history)

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    # Verifica se a requisição contém o arquivo
    if 'pdf' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    
    file = request.files['pdf']
    
    # Se o usuário não selecionou um arquivo
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    # Verifica se o arquivo tem a extensão permitida
    if file and allowed_file(file.filename):
        # Gera o nome do arquivo e salva no diretório de upload
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return jsonify({'status': 'success', 'message': 'PDF enviado com sucesso!'}), 200
    
    return jsonify({'status': 'error', 'message': 'Arquivo não permitido. Apenas PDFs são aceitos.'}), 400

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(debug=True)
