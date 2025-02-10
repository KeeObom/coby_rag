import streamlit as st
import os
import faiss
import re
from re import search
from langchain.schema import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# # Set this flag to True or False to control if the thinking process is shown in the response
# show_thinking_process = False

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If question is not in the retrieved context then say that context provided is not enough. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

pdfs_directory = "/Users/apple/Documents/ML Projects/coby/pdfs/"
faiss_db_path = "/Users/apple/Documents/ML Projects/coby/faiss_index"  # Path to store FAISS index

embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
model = OllamaLLM(model="deepseek-r1:1.5b")


# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # Stores tuples of (user_message, assistant_response)

# # Initialize chat history
# chat_history = []

# Initialize FAISS index in session state
if "vector_store" not in st.session_state:
    if os.path.exists(faiss_db_path):
        print("🔹 Loading existing FAISS index...")
        st.session_state["vector_store"] = FAISS.load_local(faiss_db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        st.session_state["vector_store"] = None  # Empty vector store until documents are uploaded


# def load_or_create_faiss():
#     if os.path.exists(faiss_db_path):
#         print("🔹 Loading existing FAISS index...")
#         return FAISS.load_local(faiss_db_path, embeddings, allow_dangerous_deserialization=True)
#     else:
#         print("🆕 No FAISS index found. Creating a new one...")
#         # Create a dummy document with page_content
#         dummy_doc = Document(page_content="This is a dummy document for index initialization.")
#         return FAISS.from_documents([dummy_doc], embeddings)  # Create new FAISS index
    

# # Try loading FAISS index
# vector_store = load_or_create_faiss()



def upload_pdf(file):
    """Save uploaded PDF to local directory."""
    file_path = os.path.join(pdfs_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def load_pdf(file_path):
    """Load pdf and extract text."""
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    return text_splitter.split_documents(documents)

# def index_docs(documents):
#     global vector_store  # Ensure we modify the existing vector_store
#     if vector_store is None:
#         vector_store = FAISS.from_documents(documents, embeddings)  # ✅ Initialize FAISS only if documents exist
#     else:
#         vector_store.add_documents(documents)  # ✅ Add new docs without overwriting
#     vector_store.save_local(faiss_db_path)  # ✅ Save persistently

def index_docs(documents):
    """Index documents into FAISS and store in session state."""
    if not documents:
        return
    
    if st.session_state["vector_store"] is None:
        st.session_state["vector_store"] = FAISS.from_documents(documents, embeddings)
    else:
        st.session_state["vector_store"].add_documents(documents)

    st.session_state["vector_store"].save_local(faiss_db_path)


def retrieve_docs(query):
    """Retrieve similar documents from FAISS."""
    if st.session_state["vector_store"] is None:
        return []
    return st.session_state["vector_store"].similarity_search(query, k=3) # Limit number of retrieved docs to 3

# def retrieve_docs(query):
#     if vector_store is None:
#         return []  # Avoid errors if no documents are indexed
#     return vector_store.similarity_search(query)


def answer_question(question, documents):
    """Generate an answer based on retrieved documents."""
    if not documents:
        return "I couldn't find relevant information in the indexed documents."

    context = "\n\n".join([doc.page_content for doc in documents])

    # Ensure answer is only based on provided context
    if not context.strip():
        return "The context provided is not enough to answer this question."

    # Build chat history for a continuous conversation feel
    conversation_history = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in st.session_state["chat_history"]])

    # Construct the full context with chat history
    full_context = f"{conversation_history}\n\nQuestion: {question}\nContext: {context}\nAnswer:"

    prompt = ChatPromptTemplate.from_template(full_context)
    chain = prompt | model

    answer = chain.invoke({"question": question, "context": context})

    return clean_text(answer)

# Remove the <think> from response
def clean_text(text: str):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

# def answer_question(question, documents):
#     """Generate an answer based on retrieved documents."""
#     if not documents:
#         return "I couldn't find relevant information in the indexed documents."
#     context = "\n\n".join([doc.page_content for doc in documents])

#     # Add chat history context for continuous conversation
#     conversation_history = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history])

#     # Construct the full context with the conversation history and the current question
#     full_context = f"{conversation_history}\n\nQuestion: {question}\nContext: {context}\nAnswer:"

#     prompt = ChatPromptTemplate.from_template(full_context)
#     chain = prompt | model

#     answer = chain.invoke({"question": question, "context": context})

#     # Optionally show thinking process
#     if show_thinking_process:
#         return f"Thinking...\n\n{answer}"
#     else:
#         return answer


# Streamlit UI
st.title("Chat with Your Documents 📄💬")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)


if uploaded_file:
    file_path = upload_pdf(uploaded_file)
    documents = load_pdf(file_path)
    chunked_documents = split_text(documents)

    if chunked_documents:  # ✅ Ensure only non-empty docs are indexed
        index_docs(chunked_documents)
        st.success(f"Document {uploaded_file.name} indexed successfully!")
    else:
        st.warning(f"No text extracted from {uploaded_file.name}. Try another file.")

# Display previous chat history
for user_msg, assistant_msg in st.session_state["chat_history"]:
    st.chat_message("user").write(user_msg)
    st.chat_message("assistant").write(assistant_msg)

# Input field for new user messages
question = st.chat_input("Ask a question...")



if question:
    st.chat_message("user").write(question)
    
    related_documents = retrieve_docs(question)
    answer = answer_question(question, related_documents)

    # Store chat history in session state
    st.session_state["chat_history"].append((question, answer))

    st.chat_message("assistant").write(answer)