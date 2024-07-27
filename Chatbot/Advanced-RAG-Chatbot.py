from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
import streamlit as st

# Load environment variables
load_dotenv()

# Load and split documents
loader = TextLoader('./materials/torontoTravelAssistant.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=300)  # Adjusted chunk size and overlap
docs = text_splitter.split_documents(documents)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Initialize Pinecone instance
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

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

# Define refined prompt template
template = """
You are a Toronto travel assistant. Users will ask you questions about their trip to Toronto. Use the following combined context to answer the question accurately and concisely.
If the combined context does not contain the answer, simply state that you don't know.

Combined Context: {context}
Question: {question}
Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Define a function for combining contexts
def merge_contexts(contexts):
    # Combine the contexts into a single string
    return "\n\n".join([ctx.page_content for ctx in contexts])

# Create a function to retrieve and merge contexts
def retrieve_and_merge_contexts(query, retriever):
    # Retrieve multiple contexts
    retrieved_docs = retriever.get_relevant_documents(query)
    # Merge contexts
    combined_context = merge_contexts(retrieved_docs)
    return combined_context

# Create a RetrievalQA chain with the refined prompt and custom retrieval
def generate_response(question):
    # Retrieve and merge context
    combined_context = retrieve_and_merge_contexts(question, docsearch.as_retriever())
    # Format the prompt
    formatted_prompt = prompt.format(context=combined_context, question=question)

    # Generate response
    response = llm.invoke(formatted_prompt)
    return response.content

st.title('Toronto Travel Assistant Bot')

# Store LLM generated responses
if "messages" not in st.session_state:
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
            with st.spinner("Getting your answer from our knowledge base..."):
                response = generate_response(input)
                st.write(response) 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
