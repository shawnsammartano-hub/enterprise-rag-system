#!/usr/bin/env python3
"""
Hybrid Search - Enterprise Best Practice
Combines semantic (vector) + lexical (BM25 keyword) search
Used by: Elasticsearch, Databricks, Pinecone, Weaviate
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.rag_pipeline import EnterpriseRAG
from typing import List, Dict, Any
from collections import Counter
import math

class HybridRAG(EnterpriseRAG):
    """Enhanced RAG with hybrid search capabilities"""
    
    def bm25_score(self, query: str, document: str, k1: float = 1.5, b: float = 0.75) -> float:
        """
        BM25 (Best Matching 25) - Industry standard keyword scoring
        
        Used by: Elasticsearch, OpenSearch, Lucene, Databricks
        
        Parameters:
        - k1: Term frequency saturation (1.2-2.0, default 1.5)
        - b: Length normalization (0-1, default 0.75)
        
        Higher score = better keyword match
        """
        # Tokenize and normalize
        query_terms = set(query.lower().split())
        doc_terms = document.lower().split()
        
        if not doc_terms:
            return 0.0
        
        # Document stats
        doc_length = len(doc_terms)
        avg_doc_length = 500  # Our chunk size
        
        # Term frequencies in document
        doc_freq = Counter(doc_terms)
        
        score = 0.0
        for term in query_terms:
            if term in doc_freq:
                tf = doc_freq[term]
                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                score += numerator / denominator
        
        return score
    
    def hybrid_query(
        self, 
        question: str, 
        n_results: int = 5, 
        alpha: float = 0.7,
        show_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Hybrid search: alpha * vector + (1-alpha) * keyword
        
        Parameters:
        - alpha: Weight for vector vs keyword (0-1)
          0.7 = 70% semantic, 30% keyword (enterprise default)
          0.5 = balanced
          0.9 = heavy semantic
        
        Why hybrid?
        - Vector search: semantic meaning, synonyms, concepts
        - Keyword search: exact terms, acronyms, IDs, names
        - Combined: best of both worlds
        """
        # Step 1: Vector search (get 2x candidates for re-ranking)
        question_embedding = self.get_embedding(question)
        vector_results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=min(n_results * 2, 20)
        )
        
        # Step 2: Calculate hybrid scores
        hybrid_scores = []
        for doc, meta, distance in zip(
            vector_results['documents'][0],
            vector_results['metadatas'][0],
            vector_results['distances'][0]
        ):
            # Vector similarity (convert distance to similarity 0-1)
            vector_score = 1 / (1 + distance)
            
            # BM25 keyword score
            bm25_score = self.bm25_score(question, doc)
            
            # Normalize BM25 (rough normalization)
            bm25_normalized = min(bm25_score / 10.0, 1.0)
            
            # Combine with alpha weighting
            hybrid_score = alpha * vector_score + (1 - alpha) * bm25_normalized
            
            hybrid_scores.append({
                'doc': doc,
                'meta': meta,
                'vector_score': vector_score,
                'bm25_score': bm25_normalized,
                'hybrid_score': hybrid_score
            })
        
        # Step 3: Re-rank by hybrid score
        hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
        top_results = hybrid_scores[:n_results]
        
        # Step 4: Build context
        context = "\n\n".join([
            f"[Source: {r['meta']['source']} - Chunk {r['meta']['chunk_index']+1}]\n{r['doc']}"
            for r in top_results
        ])
        
        # Step 5: Generate LLM response
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
        
        result = {
            "question": question,
            "answer": answer,
            "sources": [r['meta']['source'] for r in top_results],
            "num_chunks_used": len(top_results),
            "search_params": {
                "alpha": alpha,
                "n_results": n_results,
                "method": "hybrid (vector + BM25)"
            }
        }
        
        # Optionally include scoring breakdown
        if show_scores:
            result["scoring_breakdown"] = [
                {
                    "chunk_preview": r['doc'][:100] + "...",
                    "source": Path(r['meta']['source']).name,
                    "vector_score": round(r['vector_score'], 3),
                    "bm25_score": round(r['bm25_score'], 3),
                    "hybrid_score": round(r['hybrid_score'], 3)
                }
                for r in top_results
            ]
        
        return result
    
    def compare_search_methods(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Compare pure vector vs pure keyword vs hybrid
        Educational tool for understanding trade-offs
        """
        print(f"\n🔬 Comparing Search Methods\n")
        print(f"Question: {question}\n")
        
        # Pure vector (alpha=1.0)
        print("1️⃣ Pure Vector Search (Semantic Only)...")
        vector_only = self.hybrid_query(question, n_results, alpha=1.0, show_scores=True)
        
        # Pure keyword (alpha=0.0)
        print("2️⃣ Pure Keyword Search (BM25 Only)...")
        keyword_only = self.hybrid_query(question, n_results, alpha=0.0, show_scores=True)
        
        # Hybrid (alpha=0.7)
        print("3️⃣ Hybrid Search (70% Vector + 30% Keyword)...")
        hybrid = self.hybrid_query(question, n_results, alpha=0.7, show_scores=True)
        
        return {
            "question": question,
            "vector_only": vector_only,
            "keyword_only": keyword_only,
            "hybrid": hybrid
        }

# Demo
if __name__ == "__main__":
    print("🔍 Hybrid Search Demo - Enterprise RAG\n")
    
    rag = HybridRAG()
    
    # Check if data exists
    if rag.collection.count() == 0:
        print("⚠️  No documents in database. Upload documents first via the web UI.")
        exit(1)
    
    # Test hybrid search
    print("=" * 60)
    result = rag.hybrid_query(
        "What oil should I use for DR650?",
        n_results=3,
        alpha=0.7,
        show_scores=True
    )
    
    print(f"\n💡 Answer:\n{result['answer']}\n")
    print(f"📎 Sources: {', '.join([Path(s).name for s in result['sources']])}\n")
    
    if 'scoring_breakdown' in result:
        print("📊 Scoring Breakdown:")
        for i, score in enumerate(result['scoring_breakdown'], 1):
            print(f"\n  Chunk {i} ({score['source']}):")
            print(f"    Vector:  {score['vector_score']}")
            print(f"    BM25:    {score['bm25_score']}")
            print(f"    Hybrid:  {score['hybrid_score']}")
