##export KMP_DUPLICATE_LIB_OK=TRUE
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from reportlab.lib.utils import ImageReader
from PIL import Image as PILImage
from flask import Flask, render_template, request, jsonify, make_response, session
import os
import re
import io
import tempfile
import requests
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageTemplate, Frame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

import os
from datetime import datetime
from pymongo import MongoClient
from pymongo.server_api import ServerApi


load_dotenv()
app = Flask(__name__)
app.secret_key = 'cybersnow'

# MongoDB connection URI
#uri = "mongodb+srv://aimathavan14:12345678cyberm@cluster0.kns3rzh.mongodb.net/"
uri="mongodb+srv://aimathavan14:12345678cyberm@cluster0.kns3rzh.mongodb.net/"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["chat_app"]
messages_collection = db["messages"]

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

def add_message(sender, message):
    """
    Adds a new chat message to the MongoDB database.

    Args:
        sender (str): The name of the sender (either "human" or "ai").
        message (str): The content of the message.
    """
    message_data = {
        "role": sender,
        "content": message,
        "timestamp": datetime.now()
    }
    messages_collection.insert_one(message_data)
    

def get_messages():
    """
    Retrieves the latest chat messages from the MongoDB database.

    Args:
        limit (int): The maximum number of messages to retrieve.

    Returns:
        list: A list of dictionaries, where each dictionary represents a chat message.
    """
    messages = list(messages_collection.find().sort("timestamp", -1))
    
    return messages


UPLOAD_FOLDER = 'uploads'  # Folder where uploaded files will be stored
ALLOWED_EXTENSIONS = {'pdf'}  # Set of allowed file extensions
vectorstore = None
chat_history = []
pdf_summary = None
report_types=[]


# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to extract text from a PDF using Apache Tika
# Function to extract text from a PDF using Apache Tika
def extract_text_from_pdf(pdf_path):
    model = ocr_predictor(pretrained=True)
    # PDF
    doc = DocumentFile.from_pdf(pdf_path)
    result = model(doc)

    json_output = result.export()
    extracted_text = "PDF context of the company:"
    for page in json_output["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    extracted_text+= ""+word["value"] + " "


    return extracted_text

# Set the path to the service account key JSON file
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/service_account_key.json"
# Load environment variables from .env file if present


# Function to split text into chunks
def get_text_chunks(cleaned_texts_with_images):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        separator=' \n ',
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(cleaned_texts_with_images)
    #print(chunks)
    return chunks


# Function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Function to initialize a conversation chain
def get_conversation(vectorstore):
    llm = ChatOpenAI()
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain



def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Your name is Cybersnow bot,Your role is a  Research Business Analyst,Answer the user's questions based on the below context with a quantitative data which is numerical  with respect to the business context only if available if not do not makeup :\n\n{context}"),
        MessagesPlaceholder(variable_name='chat_history'),
        ("user", "{input}")
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_conversation(vectorstore)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    chat_history=get_messages()
    response = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    }) 

    return response['answer']


# Function to scrape text from a webpage
def scrape_text(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
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
    global vectorstore, chat_history,pdf_summary
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'})
    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return jsonify({'error': 'No selected file'})
    if pdf_file and allowed_file(pdf_file.filename):
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, secure_filename(pdf_file.filename))
        pdf_file.save(pdf_path)
        with open(pdf_path, 'rb') as file:
            #pdf_bytes = file.read()
            raw_text = extract_text_from_pdf(pdf_path)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            pdf_summary = oneline_text(raw_text)
            add_message("user",raw_text)
            add_message("ai",pdf_summary)
            #chat_history.append(HumanMessage(content=raw_text))
            #chat_history.append(AIMessage(content=pdf_summary))
            return jsonify({
                'success': True,
                'message': 'PDF uploaded successfully. You can now ask questions.',
                'pdf_summary': pdf_summary
            })
    else:
        return jsonify({'error': 'Invalid file format'})
    
    
def oneline_text(raw_text):
    summarization_template = """Summarize the given {text} and give a short introduction about the company and their market size customers as a business intro in 3 lines"""
    summarization_prompt = ChatPromptTemplate.from_template(summarization_template)
    summarization_chain = summarization_prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
    summarization_response = summarization_chain.invoke({"text": raw_text})
    summary = summarization_response
    return summary



@app.route('/handle_userinput', methods=['POST'])
def handle_userinput():
    global vectorstore
    user_question = request.json.get('user_question')
    
    
    if user_question:
        if vectorstore is None:
            return ("Please upload a PDF which will act as a knowledge base for the cyber snow bot")
        response = handle_pdf_question(user_question)
        
        if "i don't have access to real-time information" or "i dont know" or"I'm sorry" in response.lower() or "search" in user_question.lower():
            combined_input = f"{pdf_summary} {user_question}"
            response = handle_non_pdf_question(combined_input)
        else:
            response = handle_non_pdf_question(user_question)
    else:
        response = [{'type': 'bot', 'content': "Sorry, I didn't understand your question."}]
    add_message('user',user_question)
    add_message('ai',response)
    #chat_history.append(HumanMessage(content=user_question))
    #chat_history.append(AIMessage(content=response))

    return (response)




def handle_pdf_question(user_question):
    global vectorstore
    if vectorstore is None:
        return "Please upload a PDF which will act as a knowledge base for the cyber snow bot"
    if vectorstore:
        response = get_response(user_question)
        return response
    else:
        return "Sorry the cyber snow bot is unable to process the pdf please try again later"


# Function to summarize text
def summarize_text(texts, user_question):
    summarization_template = """Summarize the given {text} and answer it based on the given {user_question}highlight important points. Include all factual information, numbers, stats, etc. if available."""
    summarization_prompt = ChatPromptTemplate.from_template(summarization_template)
    summarization_chain = summarization_prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
    summarization_response = summarization_chain.invoke({"text": texts, "user_question": user_question})
    summary = summarization_response
    return summary





class TavilySearchAPIWrapper(RunnableLambda):
    def __init__(self):
        super().__init__(func=lambda inputs: self.run(inputs))

    def run(self, inputs):
        query = inputs.get("query")
        num_results = inputs.get("num_results", 4)
        try:
            results = self.client.search(query=query, num_results=num_results)
            return {"results": results}
        except Exception as e:
            print(f"Error during Tavily search: {e}")
            return {"error": "Failed to perform Tavily search"}


def handle_non_pdf_question(user_question):
    if vectorstore is None:
            response="Please upload a PDF which will act as a knowledge base for the cyber snow bot"
    api_wrapper = TavilySearchAPIWrapper()
    response = api_wrapper.run({"query": user_question, "num_results": 4})
    search_results = response.get("results", [])
    urls = [result['link'] for result in search_results]
    scraped_texts = [scrape_text(url) for url in urls]
    summary = summarize_text(scraped_texts, user_question)
    sum_chunks1 = get_text_chunks(summary)
    svectorstore1 = get_vectorstore(sum_chunks1)

    response = get_response(user_question)

    return response


# Function to generate text
def generate_text(texts,report_topics):
    ## Customized Prompt Template for User-Requested Topics

    summarization_template = """
    {text}

    Report should include the topics:
    User Requested: {topics}

    Using the above information which consists of the chat messages and also the extracted text, create a detailed report with a minimum of 1000 words. Ensure the report is well-structured, informative, and includes numerical data where available.

    A default Report should consist of the following topics and if there are no topics requested from the user then answer only the following topics:
    User requested topics:
        Introduction
        Business Analysis
        Market Analysis
        Founders background
        Conclusion
        And these topics too:{topics}

    If there are any additional topics requested by the user then add them too.
    The report should be answered in the below format.

    Default Format:
    Note: Only answer the topics requested by user
    <b>Introduction</b>
    • About the company:
        • Company name: [Insert company name]
        • Company description: [Describe qualitatively]
        • Market size: [Insert numerical data]
        • Growth projections: [Insert numerical data]
        • Target demographics: [Describe qualitatively]

    <b>Business Analysis</b>
    • Business model:
        • Revenue streams: [Describe qualitatively]
        • Target market size: [Insert numerical data]
        • Value proposition: [Describe qualitatively]
        • Revenue projections: [Insert numerical data]
        • Market share: [Insert numerical data]

    <b>Market Analysis</b>
        • Market size: [Insert numerical data]
        • Growth trends: [Describe qualitatively]
        • Competition analysis: [Describe qualitatively]
        • Opportunities: [Describe qualitatively]
        • Market share: [Insert numerical data]
        • Growth projections: [Insert numerical data]

    <b>Founders' Background</b>
        • Qualifications: [Describe qualitatively]
        • Experience: [Describe qualitatively]
        • Past successes: [Insert numerical data]
        • Industry recognition: [Describe qualitatively]
    
    <b>Conclusion</b>
    • Summarize key findings and provide recommendations for the company's future direction.
        • Include numerical data on growth projections, risk mitigation strategies, and performance improvement initiatives.

    Add below topics only if the user requested:

    If there are any topics user requested then add those topics and finally conclusion like above format

        <b>Go-to-Market Strategy</b>
            • Product/service launch: [Describe qualitatively]
            • Distribution channels: [Describe qualitatively]
            • Marketing tactics: [Describe qualitatively]
            • Partnerships: [Describe qualitatively]
            • Customer acquisition costs: [Insert numerical data]
            • Conversion rates: [Insert numerical data]

        <b>Customer Feedback</b>:
            • Satisfaction metrics: [Insert numerical data]
            • Retention rates: [Insert numerical data]

        <b>Risk Assessment</b>
        • Risk factors:
            • List specific risk factors and explain their potential impact.
            • Include numerical data on risk mitigation strategies and their effectiveness.
        • Regulatory Issues: [Describe qualitatively]

        <b>Performance Metrics</b>
        • Key metrics:
            • Revenue growth: [Insert numerical data]
            • Profitability: [Insert numerical data]
            • Customer acquisition cost: [Insert numerical data]
            • Market share: [Insert numerical data]
        • Benchmarking:
            • Performance gaps: [Insert numerical data]
            • Improvement targets: [Insert numerical data]

        <b>Strategic Analysis</b>
        • SWOT Analysis:
            • Strengths: [Describe qualitatively]
            • Weaknesses: [Describe qualitatively]
            • Opportunities: [Describe qualitatively]
            • Threats: [Describe qualitatively]
            • Market positioning: [Insert numerical data]
            • Competitive advantages: [Insert numerical data]

    
    """
    
    summarization_prompt = ChatPromptTemplate.from_template(summarization_template)

    
    summarization_chain = summarization_prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
    summarization_response = summarization_chain.invoke({"text": texts,"topics":report_topics})
    summary = summarization_response  # Extract the summary from the response

    return summary


@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    if request.method == 'POST':
        report_types = request.json.get('reportTypes')

        print('report:',report_types)
        
        
        # Get selected report types from the request
        

        # Extract text data from the request
        '''data = [msg.content for msg in chat_history]'''
        text_data=get_messages()
        '''text_data = '\n'.join(data)'''

        print(text_data)

        # Generate the summary
        summary = generate_text(text_data,report_types)

        # Replace <b> and </b> tags with bold text for subtitles
        subtitle_pattern = r'<b>(.*?)</b>'
        report_content = re.sub(subtitle_pattern, r'<font name="Helvetica-Bold">\1</font>', summary)

        # Generate the PDF report
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, watermark="cybersnow doc", topMargin=72, bottomMargin=72, leftMargin=72, rightMargin=72)
        content = []

        # Define custom styles for different elements
        styles = getSampleStyleSheet()
        heading_style = ParagraphStyle(
            'Heading1',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.darkblue,
            spaceAfter=12,
            fontName='Helvetica-Bold',
            leading=18
        )
        subheading_style = ParagraphStyle(
            'Heading2',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue,
            spaceAfter=8,
            fontName='Helvetica-Bold'  # Set font to bold for subtitles
        )
        normal_style = styles['BodyText']
        normal_style.leading = 16

        # Add title
        title = Paragraph("Research Report", styles['Title'])
        content.append(title)
        content.append(Spacer(1, 12))

        # Add text data
        paragraphs = report_content.split("\n")  # Split text into paragraphs
        for paragraph in paragraphs:
            if paragraph.strip():  # Skip empty paragraphs
                if paragraph.startswith('<font name="Helvetica-Bold">') and paragraph.endswith('</font>'):  # Bold subtitle
                    subtitle = Paragraph(paragraph[28:-7], subheading_style)
                else:  # Normal paragraph
                    subtitle = Paragraph(paragraph, normal_style)
                content.append(subtitle)
                content.append(Spacer(1, 6))  # Add space between paragraphs

        doc.build(content)

        # Reset buffer position to start
        buffer.seek(0)

        # Return the PDF file as a downloadable attachment
        response = make_response(buffer.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=report.pdf'

        return response



if __name__ == '__main__':
    app.run(debug=True)
