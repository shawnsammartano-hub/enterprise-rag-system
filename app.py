#!/usr/bin/env python3
"""
Fintech Company RAG Demo - AI-Powered Document Search
For SVP of Client Success Presentation
"""

import streamlit as st
import sys
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent))
from scripts.rag_pipeline import EnterpriseRAG

# Page config
st.set_page_config(
    page_title="Fintech Company RAG Demo",
    page_icon="🔍",
    layout="wide"
)

# Initialize RAG (use session state to persist)
@st.cache_resource
def get_rag():
    return EnterpriseRAG()

rag = get_rag()

# Header
st.title("🔍 Fintech Company - AI Document Search")
st.markdown("**Enterprise RAG System Demo** | Powered by Vector Search & LLM")
st.divider()

# Sidebar - Document Upload
with st.sidebar:
    st.header("📚 Document Management")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['txt', 'pdf', 'docx'],
        help="Upload TXT, PDF, or DOCX files"
    )
    
    if uploaded_file:
        # Save uploaded file
        upload_path = Path("documents") / uploaded_file.name
        upload_path.parent.mkdir(exist_ok=True)
        
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # OCR checkbox for PDFs
        use_ocr = False
        if uploaded_file.name.endswith('.pdf'):
            use_ocr = st.checkbox(
                "📷 Use OCR (for scanned PDFs)",
                help="Enable if PDF is image-based/scanned"
            )
        
        # Ingest based on file type
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                if uploaded_file.name.endswith('.txt'):
                    chunks = rag.ingest_txt(str(upload_path))
                elif uploaded_file.name.endswith('.pdf'):
                    chunks = rag.ingest_pdf(str(upload_path), use_ocr=use_ocr)
                elif uploaded_file.name.endswith('.docx'):
                    chunks = rag.ingest_docx(str(upload_path))
                
                st.success(f"✓ Ingested {chunks} chunks")
                st.rerun()  # Refresh to update stats
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.divider()
    
    # Collection stats
    st.subheader("📊 System Stats")
    total_chunks = rag.collection.count()
    st.metric("Total Chunks", total_chunks)
    st.caption(f"Embedding: {rag.embed_model}")
    st.caption(f"LLM: {rag.llm_model}")
    
    # Reset button
    if st.button("🔄 Reset Database", type="secondary"):
        # Delete persistent database
        import shutil
        persist_dir = Path.home() / ".rag_pipeline" / "chromadb"
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
        st.cache_resource.clear()
        st.rerun()

# Main area - Query interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🤔 Ask Questions")
    
    # Query input
    question = st.text_input(
        "Question",
        placeholder="What technology does Fintech Company use for data warehousing?",
        label_visibility="collapsed"
    )
    
    # Number of results
    n_results = st.slider("Context chunks to retrieve", 1, 5, 3)
    
    # Query button
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not question:
            st.warning("Please enter a question")
        elif total_chunks == 0:
            st.warning("Please upload documents first")
        else:
            with st.spinner("Searching knowledge base..."):
                result = rag.query(question, n_results=n_results)
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.success(result['answer'])
                
                # Display sources
                st.markdown("### 📎 Sources")
                for i, source in enumerate(result['sources'], 1):
                    st.caption(f"{i}. {Path(source).name}")
                
                # Store in session for context display
                st.session_state.last_result = result

with col2:
    st.subheader("ℹ️ About This Demo")
    st.markdown("""
    **This system demonstrates:**
    
    ✓ **Vector Search** (like Databricks)  
    ✓ **Semantic Embeddings** (like Snowflake Cortex)  
    ✓ **RAG Architecture** (Retrieval-Augmented Generation)  
    ✓ **Local LLM Integration** (Ollama)
    
    **Enterprise Use Cases:**
    - Client onboarding documentation
    - Product knowledge base
    - Policy & compliance docs
    - Technical documentation
    - Sales enablement materials
    """)

# Context viewer (if query was run)
if 'last_result' in st.session_state:
    st.divider()
    st.subheader("🔎 Retrieved Context")
    
    # Get the actual context chunks
    question_embedding = rag.get_embedding(st.session_state.last_result['question'])
    results = rag.collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )
    
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        with st.expander(f"Chunk {i} - {Path(meta['source']).name}"):
            st.text(doc)
            st.caption(f"Chunk {meta['chunk_index']+1}/{meta['total_chunks']}")

# Footer
st.divider()
st.caption("Fintech Company - Client Success AI Enablement | Demo by Shawn Sammartano")
