#!/usr/bin/env python3
"""
Metadata Filtering - Enterprise RAG Optimization

Like SQL WHERE clauses for vector search:
- Filter by date, author, department, document type
- Reduces search space = faster queries
- Improves relevance (search only relevant docs)

Databricks equivalent: Partition pruning in Vector Search
Snowflake equivalent: WHERE clause with vector_search()
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.rag_pipeline import EnterpriseRAG
from typing import List, Dict, Any, Optional
from datetime import datetime

class MetadataRAG(EnterpriseRAG):
    """RAG with metadata filtering capabilities"""
    
    def ingest_with_metadata(
        self,
        text: str,
        source: str,
        metadata: Dict[str, Any]
    ) -> int:
        """
        Ingest with custom metadata
        
        Common enterprise metadata:
        - document_type: "policy", "procedure", "manual", "report"
        - department: "HR", "Finance", "Engineering", "Legal"
        - created_date: ISO datetime
        - author: "jane.doe@company.com"
        - security_level: "public", "internal", "confidential"
        - version: "1.0", "2.0"
        """
        chunks = self.chunk_text(text)
        print(f"  → Created {len(chunks)} chunks from {Path(source).name}")
        
        embeddings = []
        ids = []
        metadatas = []
        documents = []
        
        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            chunk_id = f"{Path(source).stem}_chunk_{i}"
            
            # Combine base metadata with custom metadata
            chunk_metadata = {
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **metadata  # Add custom fields
            }
            
            embeddings.append(embedding)
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(chunk_metadata)
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"  ✓ Stored {len(chunks)} chunks with metadata: {list(metadata.keys())}")
        return len(chunks)
    
    def query_with_filter(
        self,
        question: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        n_results: int = 3
    ) -> Dict[str, Any]:
        """
        Query with metadata filtering
        
        Examples:
        - {"document_type": "manual"}
        - {"department": "Engineering"}
        - {"security_level": {"$in": ["public", "internal"]}}
        - {"created_date": {"$gte": "2024-01-01"}}
        """
        question_embedding = self.get_embedding(question)
        
        # Build query with optional filter
        query_params = {
            "query_embeddings": [question_embedding],
            "n_results": n_results
        }
        
        if metadata_filter:
            query_params["where"] = metadata_filter
        
        results = self.collection.query(**query_params)
        
        # Build context
        context = "\n\n".join([
            f"[Source: {meta['source']}]\n{doc}"
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ])
        
        # Generate answer
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
        
        import requests
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
            "filter_applied": metadata_filter or "None",
            "sources": [meta['source'] for meta in results['metadatas'][0]],
            "metadata": results['metadatas'][0]
        }

# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("METADATA FILTERING - Enterprise Search Optimization")
    print("=" * 70)
    print()
    
    rag = MetadataRAG()
    
    # Example: Ingest documents with metadata
    print("📚 Sample Ingestion with Metadata:\n")
    
    doc1 = "Fintech Company uses Databricks for data analytics and Snowflake for data warehousing."
    
    rag.ingest_with_metadata(
        text=doc1,
        source="documents/tech_stack.txt",
        metadata={
            "document_type": "technical",
            "department": "Engineering",
            "created_date": "2025-02-14"
        }
    )
    
    print("\n" + "-" * 70)
    print("\n🔍 Query Examples:\n")
    
    # Query 1: No filter
    print("1️⃣ No Filter (search all documents):")
    result1 = rag.query_with_filter("What does Fintech Company use for analytics?")
    print(f"   Answer: {result1['answer']}")
    print(f"   Filter: {result1['filter_applied']}")
    print()
    
    # Query 2: With filter
    print("2️⃣ With Filter (document_type = 'technical'):")
    result2 = rag.query_with_filter(
        "What does Fintech Company use for analytics?",
        metadata_filter={"document_type": "technical"}
    )
    print(f"   Answer: {result2['answer']}")
    print(f"   Filter: {result2['filter_applied']}")
    print()
    
    print("-" * 70)
    print("\n📝 Enterprise Metadata Patterns:")
    print("\n  Time-based filtering:")
    print('    {"created_date": {"$gte": "2024-01-01"}}')
    print("\n  Multi-value filtering:")
    print('    {"security_level": {"$in": ["public", "internal"]}}')
    print("\n  Compound filters:")
    print('    {"$and": [{"dept": "HR"}, {"type": "policy"}]}')
    print("\n  Databricks equivalent:")
    print('    WHERE department = "Engineering" AND created_date >= "2024-01-01"')
