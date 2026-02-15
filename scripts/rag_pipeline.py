#!/usr/bin/env python3
"""Enterprise RAG Pipeline - FIS Global Style"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from docx import Document as DocxDocument

class EnterpriseRAG:
    def __init__(self, collection_name: str = "fis_documents"):
        """Initialize RAG pipeline with ChromaDB"""
        self.collection_name = collection_name
        self.ollama_base = "http://localhost:11434"
        self.embed_model = "nomic-embed-text"
        self.llm_model = "llama3.1-32k"
        
        # Initialize ChromaDB with persistent storage
        persist_dir = Path.home() / ".rag_pipeline" / "chromadb"
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✓ Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "FIS Global document embeddings"}
            )
            print(f"✓ Created new collection: {collection_name}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings from Ollama"""
        response = requests.post(
            f"{self.ollama_base}/api/embeddings",
            json={"model": self.embed_model, "prompt": text}
        )
        return response.json()["embedding"]
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Smart chunking with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk)
        
        return chunks
    
    def ingest_txt(self, txt_path: str) -> int:
        """Ingest TXT document"""
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self._ingest_text(text, txt_path)
    
    def ingest_pdf(self, pdf_path: str, use_ocr: bool = False) -> int:
        """
        Ingest PDF document with optional OCR
        use_ocr=True for scanned PDFs (image-based)
        use_ocr=False for text-based PDFs (default)
        """
        reader = PdfReader(pdf_path)
        text = ""
        
        # Try extracting text first
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        # If no text found and OCR requested
        if len(text.strip()) < 50 and use_ocr:
            print(f"  ⚠ Little/no text found, using OCR...")
            text = self._ocr_pdf(pdf_path)
        
        # If still no text, raise error
        if len(text.strip()) < 50:
            raise ValueError(
                f"No text extracted from {pdf_path}. "
                f"Try use_ocr=True for scanned PDFs."
            )
        
        return self._ingest_text(text, pdf_path)
    
    def _ocr_pdf(self, pdf_path: str) -> str:
        """OCR for scanned PDFs"""
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            text = ""
            for i, image in enumerate(images):
                print(f"    OCR page {i+1}/{len(images)}...")
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n"
            
            return text
        except ImportError:
            raise ImportError(
                "OCR requires: pip install pytesseract pdf2image pillow"
            )
        except Exception as e:
            raise RuntimeError(f"OCR failed: {str(e)}")
    
    def ingest_docx(self, docx_path: str) -> int:
        """Ingest DOCX document"""
        doc = DocxDocument(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return self._ingest_text(text, docx_path)
    
    def _ingest_text(self, text: str, source: str) -> int:
        """Internal: Chunk, embed, and store text"""
        chunks = self.chunk_text(text)
        print(f"  → Created {len(chunks)} chunks from {Path(source).name}")
        
        embeddings = []
        ids = []
        metadatas = []
        documents = []
        
        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            chunk_id = f"{Path(source).stem}_chunk_{i}"
            
            embeddings.append(embedding)
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"  ✓ Stored {len(chunks)} chunks with embeddings")
        return len(chunks)
    
    def query(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """Query the RAG system"""
        question_embedding = self.get_embedding(question)
        
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results
        )
        
        context = "\n\n".join([
            f"[Source: {meta['source']}]\n{doc}"
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ])
        
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
        
        llm_response = requests.post(
            f"{self.ollama_base}/api/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        answer = llm_response.json()["response"]
        
        return {
            "question": question,
            "answer": answer,
            "sources": [meta['source'] for meta in results['metadatas'][0]],
            "num_chunks_used": len(results['documents'][0])
        }
    
    def stats(self):
        """Show collection statistics"""
        count = self.collection.count()
        print(f"\n📊 Collection Stats:")
        print(f"  Collection: {self.collection_name}")
        print(f"  Total chunks: {count}")
        print(f"  Embedding model: {self.embed_model}")
        print(f"  LLM model: {self.llm_model}")
