from flask import Flask, jsonify
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, world!"

@app.route('/api/data')
def get_data():
    data = {"name": "John Doe", "age": 30}
    return jsonify(data)

@app.route('/api/ai-pdf')
def get_data2():
    data = {"name": "John Doe", "age": 30, "test": ["this", "fuch you all", "you", "all motherfuckers get the fuck out now!"]}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
