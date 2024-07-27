from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from langchain_openai import ChatOpenAI
import streamlit as st
import json

# Load and split documents
loader = TextLoader('./materials/torontoTravelAssistant.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
docs = text_splitter.split_documents(documents)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Initialize Pinecone instance
pc = Pinecone(api_key= os.getenv('PINECONE_API_KEY'))

index_name = "langchain-demo"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )            
    )
index = pc.Index(index_name)
docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

# Initialize ChatOpenAI
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name, organization='org-G8UtpAEtkeLatwCgEhQGaPOw')


# Define prompt template
template = """
You are a Toronto travel assistant. Users will ask you questions about their trip to Toronto. Use the following context to answer the question accurately and concisely.
If you don't know the answer, just say you don't know.
Your answer should be short and concise, no longer than 2 sentences.

Context: {context}
Question: {question}
Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm, retriever=docsearch.as_retriever(), chain_type_kwargs={"prompt": prompt}
)


st.title('Toronto Travel Assistant Bot')

# Function for generating LLM response
def generate_response(input):
    result = rag_chain.invoke(input)
    return result

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm TravelBot. How can I assist you with your travel plans today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer from mystery stuff.."):
            response = generate_response(input) 
            result_text = response["result"]
            st.write(result_text) 
    message = {"role": "assistant", "content": response["result"]}
    st.session_state.messages.append(message)
