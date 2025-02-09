#import Libraries

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import langchain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader




#load the environment variables
load_dotenv()
inference_api_key = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "multi-PDF-Chatbot"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")




# Function to delete the existing FAISS index files 
# In the "index_path" below enter the file path location where "fiass_db" folder is created 
def delete_existing_faiss_index(index_path='F:\\Gen_AI_Projects\\Multi_PDF_RAG_Chatbot\\faiss_db'):

    faiss_index_file = 'F:\\Gen_AI_Projects\\Multi_PDF_RAG_Chatbot\\faiss_db\\index.faiss'
    faiss_metadata_file = 'F:\\Gen_AI_Projects\\Multi_PDF_RAG_Chatbot\\faiss_db\\index.pkl'
    print(faiss_index_file)
    print(faiss_metadata_file)

    if os.path.exists(faiss_index_file):
        os.remove(faiss_index_file)
        print(f"Deleted existing FAISS index file: {faiss_index_file}")

    if os.path.exists(faiss_metadata_file):
        os.remove(faiss_metadata_file)
        print(f"Deleted existing FAISS metadata file: {faiss_metadata_file}")




#load the pdf file
def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text




#split the text into chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks



# Load the text-embedding-004 model
embedding_model = HuggingFaceInferenceAPIEmbeddings(api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2")



#load the Vector database
def vector_store(text_chunks): 
    
    # Create FAISS index
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)
    vector_store.save_local("faiss_db")

   
    



#define chain to use tools
def get_conversational_chain(tools,ques):
    
    llm=ChatGroq(groq_api_key=groq_api_key,model="mixtral-8x7b-32768",temperature=0)


    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ You are a helpful assistant. Answer the question as detailed as possible and it should be based solely on the provided context from the vector database, make sure Always prioritize accuracy and relevance to the context provided.
            if the answer is not in provided context just say, "answer is not available in the context". Do not make a tool call to answer the user's question or do not answer from general knowledge, answer only if the information is readily available in the provided context. """
            ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
    tool=[tools]

    agent = create_tool_calling_agent(llm, tool, prompt)
    
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    
    response=agent_executor.invoke({"input": ques})
    
    output = response['output']
    print("output: ",output)
    
    return output




#define user input
def user_input(user_question):
    
    new_db = FAISS.load_local("faiss_db", embedding_model,allow_dangerous_deserialization=True)
    
    retriever=new_db.as_retriever(search_type="similarity",search_kwargs={"k":3})
    retrieval_chain= create_retriever_tool(retriever,"pdf_extractor","This tool is to give answer to queries from the pdf and if muliple pdf is uploaded do not combine the answers from multiple pdfs and give response only based on the one pdf which is relevant to the user query.")
    output = get_conversational_chain(retrieval_chain,user_question)
    return output




#define main function
def main():
    st.set_page_config("Chat PDF")

    
    st.title("💬RAG based Chat with PDF")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Ask a Question from the PDF Files"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        print(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = user_input(prompt)
        print("executed")
        print(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)




    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done")



        if st.button('Reset & create new chatbot'):
            delete_existing_faiss_index(index_path='F:\\Gen_AI_Projects\\Multi_PDF_RAG_Chatbot\\faiss_db')
            st.write("PDF data has been reset. now you can upload new pdfs and create a new chatbot")



if __name__ == "__main__":
    main()

