import gradio as gr
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

# Autenticate user
def authentication(username, password):
    users = {"CS50": "CS50", "user1": "password1"}
    if username in users and password == users[username]:
        return True
    else:
        return False
# Read he pdf file and return a list
def read_pdf(pdf_file):
    pdf_text = ''
    for i, page in enumerate(pdf_file.pages):
        text = page.extract_text()
        if text:
            pdf_text += text

    return pdf_text

# Split the list into smaller chuncks of texts
def split_text(pdf_text):

    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )

    return text_splitter.split_text(pdf_text)

# Pass the function into the aimodel
def aimodel(file, text, api_key):
    #connect to the api_key inputed
    os.environ["OPENAI_API_KEY"] = api_key

    #PDF processings
    pdf_file = PdfReader(file)
    pdf_text = read_pdf(pdf_file)
    texts = split_text(pdf_text)

    #Embed the text for the ai model and pass it to the Langchin pipline
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    #process the user input
    query = text
    docs = docsearch.similarity_search(query)
    return chain.run(input_documents=docs, question=query)

def main(file, text, api_key):
    #try to catch any invalid file inputs or invalid api_keys
    try:
        with open(file.name, 'rb') as f:
            result = aimodel(f, text, api_key)
        return result
    except Exception:
        return "Invalid PDF file/Open AI key. Please upload a valid PDF file or Open AI."


# The user interface
api_key = gr.inputs.Textbox(label="Enter OpenAI API Key")

gr.Interface(
    fn=main,
    inputs=[
        gr.inputs.File(label="Upload PDF file ", type="file"),
        gr.inputs.Textbox(label="What is your question?"),
        api_key
    ],
    outputs="text",
    title="PDFGPT",
    description="Upload a file and enter text to prompt PDFGPT"
).launch(auth=authentication)
