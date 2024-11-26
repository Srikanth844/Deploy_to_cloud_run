import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from typing import List

from langchain.document_loaders import (
    UnstructuredHTMLLoader,
    PyPDFLoader,
    CSVLoader,
    TextLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

app = Flask(__name__)

# Ensure the OPENAI_API_KEY is set in your Cloud Run environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key"

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'html', 'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables
chain = None
docsearch = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(file_path):
    file_type = file_path.split('.')[-1].lower()
    
    if file_type == 'txt':
        loader = TextLoader(file_path)
    elif file_type == 'html':
        loader = UnstructuredHTMLLoader(file_path)
    elif file_type == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_type == 'csv':
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = loader.load_and_split(text_splitter)
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(documents, embeddings)

@app.route('/upload', methods=['POST'])
def upload_file():
    global chain, docsearch
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            docsearch = process_file(file_path)
            
            message_history = ChatMessageHistory()
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                chat_memory=message_history,
                return_messages=True,
            )

            chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(model_name="gpt-4", temperature=0),
                chain_type="stuff",
                retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
                memory=memory,
                return_source_documents=True,
            )
            
            return jsonify({"message": f"File {filename} processed successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/query', methods=['POST'])
def query():
    global chain
    
    if not chain:
        return jsonify({"error": "No file has been processed yet"}), 400
    
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        result = chain({"question": data['question']})
        answer = result["answer"]
        source_documents = result["source_documents"]  # type: List[Document]

        sources = []
        for idx, doc in enumerate(source_documents):
            sources.append(f"Source {idx + 1}: {doc.page_content[:100]}...")

        response = {
            "answer": answer,
            "sources": sources
        }
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "LLM Query Service is running. Use /upload to process a file and /query to ask questions."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)