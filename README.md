# Calculating and Reporting Metrics of the RAG Pipeline

## Overview
The aim of this project is to evaluate the performance of the RAG pipeline and explore methods to enhance its metrics. This project includes a Python notebook and a report file, which document the evaluation process and present an improved version of our RAG chatbot. 
We utilize our Simple-RAG-Chatbot for the RAG pipeline in this project. You can find the project in this repository: [Simple-RAG-ChatBot](https://github.com/Faridghr/Simple-RAG-Chatbot)

## Project Structure
- **Chatbot/**: Contains **Simple** and **Advanced** RAG chatbot python scripts.
- **Chatbot/materials/**: Contains data that our model will use to answer questions.
- **Notebook/**: Contain a Python notebook, which document the evaluation process.
- **report/**: Stores [Report](report) files.
- **video/**: Contain [video](video) presentation. You can also watch the video on [YouTube](https://youtu.be/1D7eWNzOfqY).
- **.env**: Contains API keys.

## Dependencies
- Python 3.7+
- langchain
- pinecone-client
- python-dotenv
- streamlit
- pypdf
- scikit-learn 
- rouge-score

## Usage Evaluation Notebook file
1. Clone the repository: `git clone https://github.com/Faridghr/Evaluate-RAG-Pipeline.git`
2. Navigate to the project directory: `cd Evaluate-RAG-Pipeline`
3. Install dependencies: `pip install -r requirements.txt`
4. Set up your LLM.
5. Set up your Pinecone API key in `.env` file.
5. Navigate to src directory: `cd Notebook`
6. Run the Notebook.

## Setting Up OpenAI API
1. Enter our OpenAI account and navigate to [OpenAI Platform](https://platform.openai.com/apps). 
2. Navigate to the API section.
3. Proceed to create a new API key by pressing '+ Create' new secret key.
4. Select a suitable name to remember and press the Create secret key button.
5. Copy the secret key and add your OpenAI API Keys in a file called `.env`.

## Setting up Pinecone
1. To create a PineCone account, sign up via this link: [Pinecone](https://www.pinecone.io/)
2. After registering with the free tier, go into the project, and click on Create a Projec.
3. Fill in the Project Name, Cloud Provider, and Environment. In this case, I have used “SimpleRAGChatbot Application” as a Project Name, GCP as Cloud Provider, and Iowa (gcp-starter) as an Environment.
4. After the project is created, go into the API Keys section, and make sure you have an API key available. Do not share this API key.
5. After completing the account setup, you can add your Pinecone API Keys in a file called `.env`.