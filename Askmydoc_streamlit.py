
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from transformers import pipeline
from huggingface_hub import login
from langchain_ollama.llms import OllamaLLM
from langchain.chains import LLMChain
import chromadb
import base64
import re
import os

# Process the uploaded file.
def save_uploaded_file(uploaded_file):
    # Define a folder to save the file temporarily
    temp_dir = "uploaded_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Provide the directory path to save the file.
    file_path = os.path.join(temp_dir, uploaded_file.name)

    # Save the file to the directory
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

# Function to load PDF file
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# Function to load Text file
def load_txt(txt_path):
    loader = TextLoader(txt_path)
    documents = loader.load()
    return documents

# Function to load Word docx file
def load_word(word_path):
    loader = UnstructuredWordDocumentLoader(word_path)
    documents = loader.load()
    return documents

# Load data based on the file extension.
def load_data(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        return load_pdf(file_path)
    elif ext == 'txt':
        return load_txt(file_path)
    elif ext == 'docx':
        return load_word(file_path)
    else:
        raise ValueError("Unsupported file type")

# Perform Sentiment Analysis

# Create a pipeline to analyze the sentiment of user input.
sentiment_analyzer = pipeline('sentiment-analysis')
def analyze_sentiment(question):
    sentiment = sentiment_analyzer(question)[0]
    return sentiment['label']  # e.g., 'POSITIVE', 'NEGATIVE', 'NEUTRAL'


# Streamlit UI

# Convert image to base64 and create URL to use as background
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode()
    return img_b64

image_path = "/Users/kundhana_hp/Downloads/hackathon/chitti_the_robo.jpg"
img_b64 = image_to_base64(image_path)
image_url = f"data:image/jpg;base64,{img_b64}"
# Some CSS for styling
st.markdown(
    f"""
    <style>
    body {{
        background-image: url("{image_url}");
        background-size: 70%;
        background-position: right center;
        background-repeat: no-repeat;
        height: 100vh;  
        margin: 0;  
        overflow: hidden; 
        padding-left: 30px;
        padding-right: 10px;
    }}
    .reportview-container {{
        background: transparent;  
        height: 100vh;  
    }}
    .css-1v0mbdj, .css-1v0mbdj > div {{
        display: flex;
        flex-direction: column;
        justify-content: center;  
        align-items: flex-start;  
        height: 100vh;
        padding-left: 30px;  
    }}
    .stApp {{
        background: transparent;  
    }}
    .stTitle {{
        text-align: left;  
        padding-left: 20px;  
    }}
    .stFileUploader, .stButton, .stTextInput, .stSelectbox {{
        display: flex;
        justify-content: flex-start;  
        margin-left: 20px;  
    </style>
    """,
    unsafe_allow_html=True
)
st.title('AskMyDoc: Upload, Summarize, and Ask Anything')

# Upload document
uploaded_file = st.file_uploader("Upload a PDF, TXT, or DOCX file", type=["pdf", "txt", "docx"])
st.session_state.summary = ""
# Call appropriate fns listed above to process the uploaded file.
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    docs = load_data(file_path)
    st.write(f"Document loaded: {uploaded_file.name}")
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)
    

# Replace 'my_huggingface_token' with your token
    login('my_huggingface_token')

# Initialize embeddings and vector store
# Configure Chroma to use a remote database (cloud-based)
# Ensure a local directory to store the vector store data

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    chroma_db_dir = "chroma_db"  # Directory for local Chroma storage
    if not os.path.exists(chroma_db_dir):
        os.makedirs(chroma_db_dir)

    # Initialize Chroma vector store with local storage
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=chroma_db_dir  # Specify the directory for local storage
    )

    # Set up the retriever from the vector store
    retriever = vectorstore.as_retriever()

    # Define the prompt template with sentiment input
    new_prompt_template = """You are an expert communicator with a talent for delivering seamless, cohesive, and engaging explanations. 
    Review the context carefully and provide a detailed, single-paragraph response that is articulate, formal, and flows smoothly. 
    Avoid using bullet points, line breaks, or any formatting that interrupts the narrative. Present the information in a continuous and refined 
    manner, maintaining clarity and professionalism throughout.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    SENTIMENT:
    {sentiment}

    ANSWER:
    """
    
    prompt = PromptTemplate(template=new_prompt_template, input_variables=["context", "question", "sentiment"])

    # Load your LLM model that runs locally in ollama
    llm = OllamaLLM(model="llama3.2")
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    

# Retrieve the Context and summarize
    
    summary = """Please provide a formal, concise, and structured summary from documents. Ensure that the summary is clear, accurate, without any unnecessary details. 
        The summary should focus on the main points, key arguments, and relevant information while maintaining clarity and formality. 
         Avoid subjective and emotive language' The summary should be objective and free of unnecessary details in 200 words. Avoid words like unfortunately, happy, apologize. 
     """
# Manage session state and store summary
    if "Summary" not in st.session_state.summary:

        sentiment_result1 = analyze_sentiment(summary)
        context1 = "\n\n".join(doc.page_content for doc in retriever.get_relevant_documents(summary))


                # Run the RAG Chain with the Question, Context, and Sentiment
        s_result = llm_chain.run({
                    "context": context1,
                    "question": summary,
                    "sentiment": sentiment_result1
                })
        st.session_state.summary = "Summary: "+s_result 
    st.write(f" {st.session_state.summary}")


# Capture user input for the question
    user_question = st.text_input("Please ask a question:")
    
# Security check
    if ("password" in user_question.lower() or 
        "phone number" in user_question.lower() or 
        "ssn" in user_question.lower()):
        st.write(
            "I apologize, but I'm unable to provide any phone numbers/SSNs/passwords. "
            "As an AI question and answer tool, I am designed to assist with general queries. "
            "In the documents I process, sensitive information like passwords may appear, "
            "but I prioritize maintaining privacy and security in all responses."
        )
    else:
        if user_question:
            # Perform Sentiment Analysis on the Question
            sentiment_result = analyze_sentiment(user_question)
            
            # Retrieve the Context 
            context = "\n\n".join(doc.page_content for doc in retriever.get_relevant_documents(user_question))

            # Run the RAG Chain with the Question, Context, and Sentiment
            result = llm_chain.run({
                "context": context,
                "question": user_question,
                "sentiment": sentiment_result
            })
            
            # Display the Answer
            st.write(f"Answer: {result}")
