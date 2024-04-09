![image](https://github.com/CyberSnowTeam/PitchDeck/assets/121884337/3ceddf14-c2db-4cc4-9008-95dedb9816a2)
# Pitch Deck PDF Analysis Chatbot

## Project Overview:
This project implements a chatbot capable of analyzing pitch deck PDF files uploaded by users. The chatbot extracts text from the PDFs, performs text analysis, and engages in conversation with users based on their queries regarding the pitch deck content.

## Technologies Used:
- **Flask:** Flask is a micro web framework for Python used to develop web applications.
- **Pytesseract:** Pytesseract is a Python wrapper for Google's Tesseract-OCR Engine, used for optical character recognition (OCR).
- **pdf2image:** This library is used to convert PDF files to images, facilitating text extraction from PDFs.
- **langchain:** langchain is a library for conversational AI and natural language processing tasks. It provides tools for text splitting, embeddings, vector stores, and conversation management.
- **OpenAIEmbeddings:** OpenAIEmbeddings is used for generating embeddings of text chunks.
- **FAISS:** FAISS is a library for efficient similarity search and clustering of dense vectors.
- **DuckDuckGoSearchAPIWrapper:** DuckDuckGoSearchAPIWrapper is utilized for fetching search results from DuckDuckGo search engine.
- **dotenv:** dotenv is used to load environment variables from a .env file.
- **Werkzeug:** Werkzeug is a WSGI (Web Server Gateway Interface) utility library for Python, providing necessary functionalities for handling file uploads.

## Setup Instructions:
1. Install Python and Flask if not already installed.
2. Install Tesseract-OCR and add its path to the system environment variables.
3. Clone the project repository
4. Navigate to the project directory.
5. Install the required Python packages by running: `pip install -r requirements.txt`.
6. Create a `.env` file and define the necessary environment variables.
7. Run the Flask application by executing `python app.py`.
8. Access the chatbot interface through a web browser.

## Usage:
- Visit the home route of the Flask application to access the chatbot interface.
- Upload a pitch deck PDF file for analysis.
- The chatbot will extract text from the PDF and initialize a conversation with the user.
- Users can ask queries related to the pitch deck content.
- The chatbot will respond based on the analyzed content of the uploaded pdf.
