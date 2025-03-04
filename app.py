import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import MessagesPlaceholder
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("📄 AI-Powered PDF Q&A")
st.write("Upload a PDF and ask questions to extract insights.")

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

groq_api_key = os.getenv("GROQ_API_KEY")

# Function to extract text from PDFs
def extract_text_from_pdf(uploaded_files):
    all_text = ""
    for uploaded_file in uploaded_files:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        all_text += "\n".join([page.get_text() for page in doc]) + "\n"
    return all_text

# Function to process PDF files and answer queries
def process_pdfs_and_query(uploaded_files, query):
    if uploaded_files:
        pdf_contents = extract_text_from_pdf(uploaded_files)
        
        # Split the text into chunks
        text_splitter = CharacterTextSplitter(separator = "\n",chunk_size = 800,chunk_overlap  = 200,length_function = len)
        split_texts = text_splitter.split_text(pdf_contents)

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_texts(split_texts, embeddings)

        retriever = db.as_retriever()
        model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

        # Contextualization prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

        # System prompt for answering questions
        system_prompt = (
            "You are an AI assistant that helps summarize and answer questions from documents.\n\n"
            "Context:\n{context}\n\n"
            "Chat History:\n{chat_history}\n\n"
            "User Question:\n{input}"
        )

        qa_prompt = ChatPromptTemplate.from_template(system_prompt)

        question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        chat_history = []
        response = rag_chain.invoke({"input": query, "chat_history": chat_history})

        return response['answer']

# Streamlit File Uploader
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# User Query Input
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if uploaded_files and query:
        with st.spinner("Processing..."):
            answer = process_pdfs_and_query(uploaded_files, query)
            st.success("Answer:")
            st.write(answer)
    else:
        st.warning("Please upload at least one PDF and enter a query.")