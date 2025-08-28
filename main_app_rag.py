import streamlit as st
import os
import time
from groq import Groq

# Importar componentes de Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="LLM con RAG y Groq",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS VISUALES ---
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)


# --- FUNCIONES CORE ---

@st.cache_resource
def load_embedding_model():
    """Carga el modelo de embeddings desde Hugging Face."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_documents(uploaded_files, temp_dir="temp_docs"):
    """
    Procesa los archivos subidos (PDF, TXT, imágenes) y los divide en chunks.
    Retorna los chunks de texto listos para ser procesados por el modelo de embeddings.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    documents = []
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_path)
        else: # Para .txt, .png, .jpg, etc.
            loader = UnstructuredFileLoader(temp_path, mode="single", strategy="fast")
        
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = text_splitter.split_documents(documents)
    return docs_split

def create_vector_store(docs_split, embedding_model):
    """Crea y retorna una base de datos vectorial (FAISS) a partir de los chunks."""
    return FAISS.from_documents(docs_split, embedding_model)


# --- INTERFAZ DE STREAMLIT ---

st.title("🤖 Comparador de LLM: Respuesta General vs. con RAG")
st.markdown("Esta aplicación te permite comparar las respuestas de un LLM con y sin acceso a tus documentos (Retrieval-Augmented Generation).")

# --- SIDEBAR PARA CONFIGURACIÓN ---
with st.sidebar:
    st.header("🛠️ Configuración")
    
    # Input para la API Key de Groq
    groq_api_key = st.text_input("Ingresa tu API Key de Groq", type="password")
    
    # Selección del modelo de Groq
    st.session_state.model_name = st.selectbox(
        "Elige un modelo",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
        index=0 # Llama3 8b por defecto
    )

    st.divider()

    # Carga de archivos para el RAG
    uploaded_files = st.file_uploader(
        "Carga tus documentos (PDF, TXT, JPG, PNG)", 
        accept_multiple_files=True,
        type=['pdf', 'txt', 'jpg', 'jpeg', 'png']
    )

    if uploaded_files:
        if st.button("Procesar Documentos"):
            with st.spinner("Procesando documentos... ⏳"):
                # Cargar el modelo de embeddings una sola vez
                embedding_model = load_embedding_model()
                
                # Procesar y crear el vector store
                docs_split = process_documents(uploaded_files)
                st.session_state.vector_store = create_vector_store(docs_split, embedding_model)
                st.success("¡Documentos procesados y listos para el RAG! ✅")
    
    st.divider()
    st.header("📖 ¿Qué es un LLM y RAG?")
    st.info(
        "**LLM (Large Language Model):** Es un modelo de IA entrenado con una cantidad masiva de texto para entender y generar lenguaje humano.\n\n"
        "**RAG (Retrieval-Augmented Generation):** Es una técnica que potencia al LLM dándole acceso a información externa (tus documentos). El sistema primero 'busca' la información relevante en tus archivos y luego la usa para generar una respuesta mucho más precisa y contextualizada."
    )


# --- ÁREA PRINCIPAL ---
if not groq_api_key:
    st.warning("Por favor, ingresa tu API Key de Groq en la barra lateral para continuar.")
else:
    # Inicializar el LLM de Groq
    try:
        llm = ChatGroq(temperature=0.2, groq_api_key=groq_api_key, model_name=st.session_state.model_name)
    except Exception as e:
        st.error(f"Error al inicializar el modelo de Groq: {e}")
        st.stop()
        
    # Input para la pregunta del usuario
    prompt = st.chat_input("Haz tu pregunta aquí...")

    if prompt:
        st.markdown(f"### 💬 Tu pregunta:")
        st.info(f"**{prompt}**")

        col1, col2 = st.columns(2)

        # --- COLUMNA 1: RESPUESTA DEL LLM SIMPLE ---
        with col1:
            st.subheader("🤖 LLM (Respuesta General)")
            with st.spinner("Pensando..."):
                start_time = time.time()
                
                # Prompt simple para el LLM general
                simple_prompt = ChatPromptTemplate.from_template(
                    """
                    Responde la siguiente pregunta de la manera más concisa posible.
                    Pregunta: {question}
                    """
                )
                chain_simple = simple_prompt | llm
                response_simple = chain_simple.invoke({"question": prompt})
                
                end_time = time.time()
                response_time = end_time - start_time

            st.markdown(response_simple.content)
            st.success(f"Tiempo de respuesta: {response_time:.2f} segundos")

        # --- COLUMNA 2: RESPUESTA DEL LLM CON RAG ---
        with col2:
            st.subheader("🧠 LLM con RAG (Respuesta con tus Documentos)")
            
            if "vector_store" not in st.session_state or st.session_state.vector_store is None:
                st.warning("No has subido documentos o no han sido procesados. La respuesta RAG no estará disponible.")
            else:
                with st.spinner("Buscando en tus documentos y pensando..."):
                    start_time_rag = time.time()

                    # Crear el retriever
                    retriever = st.session_state.vector_store.as_retriever()
                    
                    # Prompt para el RAG
                    rag_prompt = ChatPromptTemplate.from_template(
                        """
                        Eres un asistente experto en responder preguntas basándote únicamente en el siguiente contexto. 
                        Si el contexto no contiene la respuesta, di que no tienes la información. Sé claro y conciso.

                        Contexto:
                        {context}

                        Pregunta: {input}
                        """
                    )
                    
                    document_chain = create_stuff_documents_chain(llm, rag_prompt)
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    
                    # Invocar la cadena y obtener la respuesta y el contexto
                    response_rag = retrieval_chain.invoke({"input": prompt})
                    
                    end_time_rag = time.time()
                    response_time_rag = end_time_rag - start_time_rag

                st.markdown(response_rag["answer"])
                st.success(f"Tiempo de respuesta: {response_time_rag:.2f} segundos")

                # Mostrar el contexto utilizado
                with st.expander("Ver el contexto utilizado por el RAG"):
                    for i, doc in enumerate(response_rag["context"]):
                        st.write(f"**Fuente {i+1}:** {doc.metadata.get('source', 'Desconocido')}")
                        st.write(doc.page_content)
                        st.divider()
