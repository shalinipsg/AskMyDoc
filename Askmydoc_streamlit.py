
import streamlit as st
# from chromadb.config import Settings
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

# Function to save the uploaded file and return the file path
def save_uploaded_file(uploaded_file):
    # Define a folder to save the file temporarily
    temp_dir = "uploaded_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Create the path where the file will be saved
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

# Function to load Word file
def load_word(word_path):
    loader = UnstructuredWordDocumentLoader(word_path)
    documents = loader.load()
    return documents

# Function to load data based on the file type
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

# Sentiment Analysis setup
sentiment_analyzer = pipeline('sentiment-analysis')

# Function to analyze sentiment
def analyze_sentiment(question):
    sentiment = sentiment_analyzer(question)[0]
    return sentiment['label']  # e.g., 'POSITIVE', 'NEGATIVE', 'NEUTRAL'







# Streamlit UI


# # Optional: Display the image to ensure it's working
# st.image(image_url)

# Function to load image and return it as base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode()
    return img_b64

# Path to your local image file
image_path = "/Users/kundhana_hp/Downloads/hackathon/chitti_the_robo.jpg"

# Convert the image to base64 and create a data URL
img_b64 = image_to_base64(image_path)
image_url = f"data:image/jpg;base64,{img_b64}"

# Optional: Check the first 1000 characters of the base64 string for debugging
st.text(img_b64[:1000])  # Just for debugging, remove this in production

# Set the background image using the base64 data URL

st.markdown(
    f"""
    <style>
    body {{
        background-image: url("{image_url}");
        background-size: 70%;
        background-position: right center;
        background-repeat: no-repeat;
        height: 100vh;  /* Ensure the background covers the entire viewport height */
        margin: 0;  /* Remove any default margin */
        overflow: hidden;  /* Hide scrollbars if needed */
        padding-left: 30px;
        padding-right: 10px;
    }}
    .reportview-container {{
        background: transparent;  /* Ensure the content area is transparent so background is visible */
        height: 100vh;  /* Full height of viewport */
    }}
    # .css-18e3th9, .css-1v0mbdj {{  /* Hide the default streamlit text */
    #     display: ;
    # }}
    .css-1v0mbdj, .css-1v0mbdj > div {{
        display: flex;
        flex-direction: column;
        justify-content: center;  /* Vertically center */
        align-items: flex-start;  /* Left-align */
        height: 100vh;
        padding-left: 30px;  /* Add left padding to align content left */
    }}
    .stApp {{
        background: transparent;  /* Make sure the content area also has a transparent background */
    }}
    .stTitle {{
        text-align: left;  /* Align the title to the left */
        padding-left: 20px;  /* Add some padding to the left */
    }}
    .stFileUploader, .stButton, .stTextInput, .stSelectbox {{
        display: flex;
        justify-content: flex-start;  /* Align these widgets to the left */
        margin-left: 20px;  /* Add some margin to the left */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Optional: Display the image directly for testing
# st.image(image_url)



st.title('AskMyDoc: Upload, Summarize, and Ask Anything')

# Upload document
uploaded_file = st.file_uploader("Upload a PDF, TXT, or DOCX file", type=["pdf", "txt", "docx"])
st.session_state.summary = ""
print(st.session_state.summary)

if uploaded_file is not None:
    # Save the uploaded file and get its file path
    file_path = save_uploaded_file(uploaded_file)

    # Load the document based on its extension
    docs = load_data(file_path)
    st.write(f"Document loaded: {uploaded_file.name}")
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)

# # Replace 'your_huggingface_token' with your actual token
    login('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')
    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Configure Chroma to use a remote database (cloud-based)
# Ensure a local directory to store the vector store data
    chroma_db_dir = "chroma_db"  # Directory for local Chroma storage
    if not os.path.exists(chroma_db_dir):
        os.makedirs(chroma_db_dir)

    # Initialize Chroma vector store with local storage
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=chroma_db_dir  # Specify the directory for local storage
    )


    # vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

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

    # Load your LLM model (assuming you have set this up)
    # Example: using a placeholder LLM model from LangChain
    # llm = HuggingFaceEmbeddings(model_name="llama3.2")  # Use your LLM setup here
    llm = OllamaLLM(model="llama3.2")
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    

# Retrieve the Context (from documents loaded earlier)
    # summary="summarize from the context"
    summary = """Please provide a formal, concise, and structured summary from documents. Ensure that the summary is clear, accurate, without any unnecessary details. 
        The summary should focus on the main points, key arguments, and relevant information while maintaining clarity and formality. 
         Avoid subjective and emotive language' The summary should be objective and free of unnecessary details in 200 words. Avoid words like unfortunately, happy, apologize. 
     """

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





    # sentiment_result1 = analyze_sentiment(summary)
    # context1 = "\n\n".join(doc.page_content for doc in retriever.get_relevant_documents(summary))


    #             # Run the RAG Chain with the Question, Context, and Sentiment
    # s_result = llm_chain.run({
    #                 "context": context1,
    #                 "question": summary,
    #                 "sentiment": sentiment_result1
    #             })
    # st.write(f"Summary: {s_result}")



    # Capture user input for the question
    user_question = st.text_input("Please ask a question:")

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
            
            # Retrieve the Context (from documents loaded earlier)
            context = "\n\n".join(doc.page_content for doc in retriever.get_relevant_documents(user_question))


            # Run the RAG Chain with the Question, Context, and Sentiment
            result = llm_chain.run({
                "context": context,
                "question": user_question,
                "sentiment": sentiment_result
            })
            
            # Display the Answer
            st.write(f"Answer: {result}")
