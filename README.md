## AskMyDoc: Upload, Summarize, and Ask Anything

# INTRODUCTION
AskMyDoc is a Streamlit-based platform that allows users to upload documents (PDF, TXT, DOCX) and ask questions. It leverages LangChain, HuggingFace embeddings, sentiment analysis, and RAG architecture for document retrieval and summarization.

# ARCHITECTURE: RAG (RETREIVAL-AUGMENTED GENERATION)
The system is built using RAG architecture, which is a hybrid approach combining retrieval-based techniques with generative models such as llms. The process follows these main steps:
1.	Document Upload: Users upload a document in PDF, TXT, or DOCX format.
2.	Document Parsing & Retrival: The uploaded file is parsed to extract its text content, which is then split into smaller chunks and using vector store (Chroma), the relevant sections of the document are retrieved based on the user's query.
3.	Sentiment Analysis: Sentiment analysis is applied to the user’s question to understand the tone (positive, negative, or neutral).
4.	Answer Generation: The system uses a Large Language Model (LLM) to generate an answer by integrating the retrieved context with the sentiment of the question.
   
# MODELS USED : 
•	HuggingFace Embeddings: The model all-MiniLM-L6-v2 is used to convert text into numerical embeddings. This allows for fast and accurate searching within large datasets.
•	Ollama LLM: The llama3.2 model is used for generating detailed, formal responses to user queries. It is fine-tuned to handle complex document-based questions.
•	Sentiment Analysis: We analyze the sentiment of the user's input, and incorporate into the prompt to generate responses that match the user's tone.
•	LangChain is a framework used here with custom prompt templates to format the context, question, and sentiment together before passing it to the LLM.
•	Streamlit is used to create an interactive front-end interface. Streamlit also handles displaying the results and document summaries.
•	Chroma is used as the vector store to manage and query document embeddings allowing for fast retrieval of relevant information based on the user's question.

# SENTIMENT ANALYSIS
Sentiment analysis is a key feature of the system. It involves classifying the user's input into one of three sentiment categories: Positive, Negative, or Neutral. This classification helps tailor the tone of the answer generated by the system, ensuring that the response aligns with the emotional tone of the user's inquiry.

# PASSWORD SECURITY
The system includes a security feature that checks if user asks for the passwords, ssn, phone numbers etc in user queries and responds with a predefined message ensuring privacy and security:“I apologize, but I'm unable to provide any passwords. As an AI question and answer tool, I am designed to assist with general queries. In the documents I process, sensitive information such as passwords or phone numbers may appear, but I prioritize maintaining privacy and security in all responses”.
