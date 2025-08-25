import os 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv 
import streamlit as st 
from streamlit_chat import message

load_dotenv()

# Update environment variables
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ['ACTIVELOOP_TOKEN'] = os.getenv('ACTIVELOOP_TOKEN')
os.environ['DEEPLAKE_ACCOUNT_NAME'] = os.getenv('DEEPLAKE_ACCOUNT_NAME')

@st.cache_data
def doc_preprocessing():
    print("**********in doc_preprocessing**********")
    docs = []
    pdf_path = 'data/Q4-2022-Amazon-Earnings-Release.pdf'
    loader = PyPDFLoader(pdf_path)
    docs.extend(loader.load())
    
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

@st.cache_resource
def embeddings_store():
    print("**********in embeddings_store**********")
    embeddings = AzureOpenAIEmbeddings(
        deployment="text-embedding-3-small",
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version="2023-05-15",
        chunk_size=1,
        validate_base_url=False
    )
    print("**********embeddings**********",embeddings)
    texts = doc_preprocessing()
    print("**********texts**********",texts)
    db = DeepLake.from_documents(texts, embeddings, dataset_path=f"hub://niranjan27405/text_embedding")
    print("**********db**********",db)
    db = DeepLake(
        dataset_path=f"hub://niranjan27405/text_embedding",
        read_only=True,
        embedding_function=embeddings,
    )
    return db

@st.cache_resource
def search_db():
    print("**********in search_db**********")
    db = embeddings_store()
    retriever = db.as_retriever()
    retriever.search_kwargs = {
        'distance_metric': 'cos',
        'k': 10
    }
    
    model = AzureChatOpenAI(
        deployment_name="gpt-4o",
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version="2024-02-15-preview",
        temperature=0
    )
    
    qa = RetrievalQA.from_llm(model, retriever=retriever, return_source_documents=True)
    return qa

qa = search_db()

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))

def main():
    # Initialize Streamlit app with a title
    st.title("LLM Powered Chatbot")

    # Get user input from text input
    user_input = st.text_input("", key="input")

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]
        
    # Search the database for a response based on user input and update session state
    if user_input:
        output = qa.invoke({"query": user_input})
        st.session_state.past.append(user_input)
        response = str(output["result"])
        st.session_state.generated.append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)

if __name__ == "__main__":
    main()
