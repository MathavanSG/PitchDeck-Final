from flask import Flask, render_template, request, jsonify
import os
import pytesseract
from pdf2image import convert_from_bytes

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper

from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document

from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from langchain.prompts import PromptTemplate




from dotenv import load_dotenv
import tempfile
from werkzeug.utils import secure_filename
import requests
from langchain import OpenAI
from bs4 import BeautifulSoup

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.utilities import DuckDuckGoSearchAPIWrapper

import json

import openai,sys
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)
load_dotenv()
UPLOAD_FOLDER = 'uploads'  # Folder where uploaded files will be stored
ALLOWED_EXTENSIONS = {'pdf'}  # Set of allowed file extensions

# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Set path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

conversation=None

def extract_text_from_pdf_with_images(pdf_bytes):
    # Convert PDF bytes to images
    images = convert_from_bytes(pdf_bytes)
    # Initialize an empty string to store extracted text
    extracted_text = ""
    # Loop through each image and extract text using OCR
    for img in images:
        text = pytesseract.image_to_string(img)
        extracted_text += text + "\n"
    return extracted_text

def get_text_chunks(cleaned_texts_with_images):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        separator='\n',
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(cleaned_texts_with_images)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation(vectorestore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def scrape_text(url):
    try:
        # Send a GET request to the webpage
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract all text from the webpage
            page_text = soup.get_text(separator=' ', strip=True)
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"

    



@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global conversation
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'})
    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return jsonify({'error': 'No selected file'})
    if pdf_file and allowed_file(pdf_file.filename):
        # Create a temporary directory to store the uploaded PDF
        temp_dir = tempfile.mkdtemp()
        # Save the PDF file to the temporary directory
        pdf_path = os.path.join(temp_dir, secure_filename(pdf_file.filename))
        pdf_file.save(pdf_path)
        # Read text from the uploaded PDF
        with open(pdf_path, 'rb') as file:
            pdf_bytes = file.read()
            texts_with_images = extract_text_from_pdf_with_images(pdf_bytes)
            raw_text = texts_with_images
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            conversation = get_conversation(vectorstore)  # Initialize conversation
            chat_history = [{'content': 'Extracted text from the PDF:'}, {'content': raw_text}]
            return jsonify({
                'success': True,
                'message': 'PDF uploaded successfully. You can now ask questions.',
                'chat_history': chat_history
            })
    else:
        return jsonify({'error': 'Invalid file format'})


@app.route('/handle_userinput', methods=['POST'])
def handle_userinput():
    global conversation
    user_question = request.json.get('user_question')
    if user_question:
        # Check if the user input contains keywords related to PDFs
        if 'search' in user_question.lower():
            response = handle_non_pdf_question(user_question)
        else:
            response = handle_pdf_question(user_question)
    else:
        response = [{'type': 'bot', 'content': "Sorry, I didn't understand your question."}]
    return jsonify({'messages': response})


def handle_pdf_question(user_question):
    global conversation
    # Initialize conversation if not already initialized
    if conversation is None:
        default_text = "Hello,(mention you name and welcome user to cybersnow)My name  Cyber Snow Bot welcome to the cyber snow service portal please upload a PDF document so  I can  assist you further?"
        text_chunks = get_text_chunks(default_text)
        vectorstore = get_vectorstore(text_chunks)
        conversation = get_conversation(vectorstore)
    # Retrieve the saved context from the conversation memory
    response = conversation({'question': user_question})
    chat_history = response['chat_history']
    messages = [{'type': 'user', 'content': message.content} if i % 2 == 0 else {'type': 'bot', 'content': message.content}
                for i, message in enumerate(chat_history)]
    return messages

def summarize_text(texts,user_question):
    summarization_template = """Summarize the given {text} and  answer it based on the given {user_question}highlight important points Include all factual information, numbers, stats etc if available."""
    summarization_prompt = ChatPromptTemplate.from_template(summarization_template)

    summarization_chain = summarization_prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()

    summarization_response = summarization_chain.invoke({"text": texts,"user_question":user_question})
    summary = summarization_response  # Extract the summary from the response

    return summary


def handle_non_pdf_question(user_question):
    # Perform Google search
    search_results = DuckDuckGoSearchAPIWrapper().results(user_question,num_results=4)
    urls = [result['link'] for result in search_results]
    # Scrape text from search results
    scraped_texts = [scrape_text(url) for url in urls]

    # Summarize the scraped text
    summary = summarize_text(scraped_texts,user_question)


    # Prepare messages with the summary
    messages = [{'type': 'bot', 'content': summary}]
    sum_chunks = get_text_chunks(summary)
    svectorstore = get_vectorstore(sum_chunks)
    conversation = get_conversation(svectorstore)

    return messages

    


if __name__ == '__main__':
  app.run(debug=True)