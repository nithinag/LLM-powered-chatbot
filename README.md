LLM Powered Chatbot with Streamlit

This project is a Streamlit-based chatbot that uses Azure OpenAI models and DeepLake as a vector store to enable conversational search and Q&A over documents (e.g., PDFs).

🚀 Features

Streamlit web app for interactive chat.

Document ingestion using PyPDFLoader.

Text chunking with CharacterTextSplitter.

Vector embeddings generated with Azure OpenAI Embeddings.

Storage and retrieval powered by DeepLake.

Conversational responses using Azure OpenAI GPT models.

Chat history maintained within the Streamlit session.

Flow:

User enters a question in the input box.

The chatbot retrieves relevant chunks from the document via DeepLake retriever.

The selected context is passed to Azure OpenAI GPT for response generation.

Responses are displayed in a conversational UI with history tracking.
Key packages used (see requirements.txt for full list):

langchain – Framework for LLM applications

deeplake – Vector store for embeddings

streamlit & streamlit_chat – Web UI and chat interface

pypdf – PDF parsing

python-dotenv – Environment variable management
