import streamlit as st
import os
import time
import pandas as pd
from groq import Groq

# Importaciones de Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Importaciones para visualización
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Laboratorio RAG Interactivo",
    page_icon="🔬",
    layout="wide"
)

# --- ESTILOS --- (Se mantiene la paleta de colores legible)
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; color: #333333; }
    h1, h2, h3 { color: #1a252f; }
    .stButton>button { background-color: #007bff; color: white; border-radius: 8px; border: none; padding: 10px 20px; font-weight: bold; }
    .stButton>button:hover { background-color: #0056b3; }
    .stTextInput>div>div>input { background-color: #ffffff; color: #333333; }
    .stInfo { background-color: #e7f3ff; color: #0d6efd; border-left: 5px solid #0d6efd; border-radius: 5px; padding: 1rem; }
    .stSuccess { background-color: #e6f9e6; color: #198754; border-left: 5px solid #198754; border-radius: 5px; padding: 0.5rem 1rem; }
    .stWarning { background-color: #fff8e1; color: #ffc107; border-left: 5px solid #ffc107; border-radius: 5px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- ESTADO DE LA SESIÓN ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "docs_split" not in st.session_state:
    st.session_state.docs_split = []
if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = None
if "pca_model" not in st.session_state:
    st.session_state.pca_model = None


# --- FUNCIONES CORE ---
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

def process_and_embed_docs(uploaded_files, chunk_size, chunk_overlap, temp_dir="temp_docs"):
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    documents = []
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f: f.write(uploaded_file.getvalue())
        loader = UnstructuredFileLoader(temp_path)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    st.session_state.docs_split = text_splitter.split_documents(documents)
    
    if st.session_state.docs_split:
        st.session_state.vector_store = FAISS.from_documents(st.session_state.docs_split, embedding_model)
        
        # Generar embeddings para la visualización
        st.session_state.doc_embeddings = embedding_model.embed_documents([doc.page_content for doc in st.session_state.docs_split])
        
        # Entrenar PCA
        pca = PCA(n_components=2)
        st.session_state.pca_model = pca.fit(st.session_state.doc_embeddings)
        
        return True
    return False

def visualize_embeddings(query_text):
    if not st.session_state.pca_model or not st.session_state.doc_embeddings:
        st.warning("Primero procesa los documentos para poder visualizar los embeddings.")
        return

    # Proyectar embeddings de documentos
    doc_coords = st.session_state.pca_model.transform(st.session_state.doc_embeddings)
    
    df = pd.DataFrame({
        'x': doc_coords[:, 0],
        'y': doc_coords[:, 1],
        'text': [f"Chunk {i}: {doc.page_content[:100]}..." for i, doc in enumerate(st.session_state.docs_split)],
        'type': 'Documento'
    })

    # Proyectar embedding de la consulta
    if query_text:
        query_embedding = embedding_model.embed_query(query_text)
        query_coords = st.session_state.pca_model.transform([query_embedding])
        
        query_df = pd.DataFrame({
            'x': query_coords[:, 0],
            'y': query_coords[:, 1],
            'text': f"Tu Pregunta: {query_text}",
            'type': 'Pregunta'
        })
        df = pd.concat([df, query_df], ignore_index=True)

    fig = px.scatter(
        df, x='x', y='y', color='type', hover_data='text',
        title="Visualización del Espacio de Embeddings (Reducido con PCA)",
        color_discrete_map={'Documento': 'blue', 'Pregunta': 'red'},
        symbol='type',
        symbol_map={'Documento': 'circle', 'Pregunta': 'star'}
    )
    fig.update_traces(marker=dict(size=12), selector=dict(mode='markers', type='scatter'))
    fig.update_layout(legend_title_text='Tipo')
    st.plotly_chart(fig, use_container_width=True)


# --- INTERFAZ DE STREAMLIT ---
st.title("🔬 Laboratorio Interactivo de RAG")
st.markdown("Una herramienta para entender los componentes de Retrieval-Augmented Generation a nivel de posgrado.")

# --- SIDEBAR DE CONFIGURACIÓN ---
with st.sidebar:
    st.header("🛠️ 1. Configuración General")
    groq_api_key = st.text_input("Ingresa tu API Key de Groq", type="password")
    model_name = st.selectbox("Elige un modelo", ["llama3-8b-8192", "mixtral-8x7b-32768"])
    
    st.divider()
    
    st.header("🧪 2. Parámetros del RAG")
    uploaded_files = st.file_uploader("Sube tus documentos", accept_multiple_files=True)
    
    # Sliders para parámetros de chunking
    chunk_size = st.slider("Tamaño del Chunk (chunk_size)", 200, 2000, 1000, 100)
    chunk_overlap = st.slider("Solapamiento (chunk_overlap)", 0, 500, 200, 50)
    
    if uploaded_files:
        if st.button("Procesar Documentos"):
            with st.spinner("Procesando y creando embeddings... Esto puede tardar."):
                success = process_and_embed_docs(uploaded_files, chunk_size, chunk_overlap)
                if success:
                    st.success("¡Documentos procesados exitosamente!")
                else:
                    st.error("No se pudo procesar los documentos.")

# --- PESTAÑAS PRINCIPALES ---
tab1, tab2, tab3 = st.tabs(["🤖 Comparador RAG", "🗺️ Explorador de Embeddings", "⚙️ Desglose del Proceso"])

with tab1:
    st.header("Comparación: LLM simple vs. LLM con RAG")
    
    if not groq_api_key:
        st.warning("Ingresa tu API Key de Groq para continuar.")
    elif not st.session_state.vector_store:
        st.warning("Sube y procesa documentos en la barra lateral para activar el RAG.")
    else:
        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=model_name)
        
        st.subheader("Parámetros de Recuperación")
        top_k = st.slider("Chunks a recuperar (Top-K)", 1, 10, 3)
        
        prompt = st.text_input("Haz tu pregunta aquí:", key="main_prompt")

        if prompt:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Respuesta General (sin RAG)")
                with st.spinner("Pensando..."):
                    simple_chain = llm
                    response_simple = simple_chain.invoke(prompt)
                st.markdown(response_simple.content)

            with col2:
                st.subheader("Respuesta Aumentada (con RAG)")
                with st.spinner("Buscando en documentos y pensando..."):
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={'k': top_k})
                    rag_prompt = ChatPromptTemplate.from_template("Responde la pregunta basándote únicamente en este contexto:\n\n{context}\n\nPregunta: {input}")
                    document_chain = create_stuff_documents_chain(llm, rag_prompt)
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    response_rag = retrieval_chain.invoke({"input": prompt})

                st.markdown(response_rag["answer"])
                
                with st.expander(f"Ver los {top_k} chunks recuperados (contexto)"):
                    for i, doc in enumerate(response_rag["context"]):
                        st.success(f"Chunk {i+1} (Fuente: {doc.metadata.get('source', 'N/A')})")
                        st.write(doc.page_content)

with tab2:
    st.header("Visualización del Espacio Vectorial")
    query_vis = st.text_input("Escribe una pregunta para visualizarla en el espacio vectorial:", key="vis_prompt")
    visualize_embeddings(query_vis)

with tab3:
    st.header("Anatomía de un Sistema RAG")
    st.markdown("""
    Un sistema RAG funciona en dos fases principales: **Indexación** y **Recuperación y Generación**.
    """)
    if not st.session_state.docs_split:
        st.info("Sube y procesa un documento para ver un ejemplo práctico aquí.")
    else:
        st.subheader("1. Fase de Indexación (Lo que ya hicimos)")
        st.markdown(f"**a. Carga y División (Chunking):** Tu documento se dividió en **{len(st.session_state.docs_split)} chunks**.")
        with st.expander("Ver ejemplo de un chunk"):
            st.code(st.session_state.docs_split[0].page_content)
        
        st.markdown("**b. Vectorización (Embeddings):** Cada chunk se convierte en un vector numérico que captura su significado semántico.")
        with st.expander("Ver ejemplo de un vector (primeras 10 dimensiones)"):
            st.code(str(np.array(st.session_state.doc_embeddings[0])[:10]) + "...")

        st.subheader("2. Fase de Recuperación y Generación (Lo que pasa cuando preguntas)")
        st.markdown("**a. Búsqueda de Similitud:** Tu pregunta se convierte en un vector y el sistema busca los chunks con los vectores más 'cercanos' en la base de datos vectorial (FAISS).")
        st.markdown("**b. Aumentación del Prompt:** Los chunks recuperados se insertan en un prompt junto a tu pregunta. Este es el 'contexto'.")
        st.markdown("**c. Generación:** El LLM recibe este prompt aumentado y genera una respuesta basada **exclusivamente** en la información proporcionada.")
