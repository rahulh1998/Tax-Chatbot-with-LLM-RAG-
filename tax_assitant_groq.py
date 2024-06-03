#Loading all the essential libraries
from langchain_community.document_loaders import PyPDFDirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
from dotenv import load_dotenv
import os

st.title("Tax Assistant with RAG with GROQ Inferencing")

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Loading Multiple PDF Files 
loader = PyPDFDirectoryLoader("../Tax PDF Data")
docs = loader.load()

# Transforming the documents in chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
documents = text_splitter.split_documents(docs)

# FAISS Vector Database
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents,embeddings)

# Using Opensource LLM - Llama3-8b Model
llm=ChatGroq(groq_api_key = groq_api_key,model_name="Llama3-8b-8192")

# Prompt Template
prompt = ChatPromptTemplate.from_template("""Consider you are a smart tax assistant who has very high knowledge in the latest and accurate Indian tax. 
            I want you to answer my question {input} in a very compact and precise manner within 200 words
            Give all the answers with respect to Indian tax context only. 
            Return the output in a clean and crisp answer 
            <context>
            {context}
            </context>
            Question:{input}
            """)

# Initialize a retriever object from the database
retriver = db.as_retriever()

# Create a document chain using the LLM and the prompt
document_chain = create_stuff_documents_chain(llm, prompt) 

# Create a retrieval chain using the retriever and the document chain
retrieval_chain = create_retrieval_chain(retriver,document_chain)

# Prompt the user for a question using a text input field
question = st.text_input("Enter your query..")

# Invoke the retrieval chain with the question as input
response = retrieval_chain.invoke({"input": question})

if question:
    st.write(response['answer'])
