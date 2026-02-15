#!/usr/bin/env python3
"""
Re-ranking - Enterprise RAG Quality Improvement

Two-stage retrieval (industry standard):
1. First pass: Fast vector search (retrieve 20-50 candidates)
2. Second pass: Expensive re-ranking (select top 3-5)

Re-ranking methods:
- Cross-encoder models (BERT-based, most accurate)
- LLM-based scoring (GPT/Claude as judge)
- Custom business logic (recency, authority, user history)

Used by: Cohere, Pinecone, Weaviate, Databricks
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.rag_pipeline import EnterpriseRAG
from typing import List, Dict, Any
import re

class ReRankingRAG(EnterpriseRAG):
    """RAG with re-ranking capabilities"""
    
    def relevance_score(self, query: str, document: str) -> float:
        """
        Simple relevance scoring for re-ranking
        
        In production, use:
        - Cross-encoder models (ms-marco-MiniLM, bge-reranker)
        - LLM-based scoring
        - Learned-to-rank models
        
        This is a simplified version for demonstration
        """
        query_terms = set(query.lower().split())
        doc_lower = document.lower()
        
        score = 0.0
        
        # Exact phrase match (high value)
        if query.lower() in doc_lower:
            score += 5.0
        
        # Term coverage (what % of query terms appear)
        matching_terms = sum(1 for term in query_terms if term in doc_lower)
        coverage = matching_terms / len(query_terms) if query_terms else 0
        score += coverage * 3.0
        
        # Position bonus (earlier mentions = more relevant)
        first_match_pos = float('inf')
        for term in query_terms:
            pos = doc_lower.find(term)
            if pos != -1:
                first_match_pos = min(first_match_pos, pos)
        
        if first_match_pos != float('inf'):
            # Earlier = higher score (normalized by doc length)
            position_score = 1.0 - (first_match_pos / len(doc_lower))
            score += position_score * 2.0
        
        return score
    
    def query_with_reranking(
        self,
        question: str,
        initial_results: int = 10,
        final_results: int = 3
    ) -> Dict[str, Any]:
        """
        Two-stage retrieval with re-ranking
        
        Stage 1: Vector search (fast, recall-focused)
        Stage 2: Re-rank (expensive, precision-focused)
        """
        # Stage 1: Vector search (get more candidates)
        question_embedding = self.get_embedding(question)
        vector_results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=initial_results
        )
        
        # Stage 2: Re-rank candidates
        candidates = []
        for doc, meta, distance in zip(
            vector_results['documents'][0],
            vector_results['metadatas'][0],
            vector_results['distances'][0]
        ):
            vector_score = 1 / (1 + distance)
            rerank_score = self.relevance_score(question, doc)
            
            # Combined score (50% vector, 50% rerank)
            combined_score = 0.5 * vector_score + 0.5 * (rerank_score / 10.0)
            
            candidates.append({
                'doc': doc,
                'meta': meta,
                'vector_score': vector_score,
                'rerank_score': rerank_score,
                'combined_score': combined_score
            })
        
        # Sort by combined score
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        top_candidates = candidates[:final_results]
        
        # Build context
        context = "\n\n".join([
            f"[Source: {c['meta']['source']}]\n{c['doc']}"
            for c in top_candidates
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
            "sources": [c['meta']['source'] for c in top_candidates],
            "reranking_stats": {
                "initial_candidates": initial_results,
                "final_results": final_results,
                "reranking_impact": [
                    {
                        "chunk_preview": c['doc'][:80] + "...",
                        "vector_score": round(c['vector_score'], 3),
                        "rerank_score": round(c['rerank_score'], 3),
                        "combined": round(c['combined_score'], 3)
                    }
                    for c in top_candidates
                ]
            }
        }

# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("RE-RANKING DEMONSTRATION - Two-Stage Retrieval")
    print("=" * 70)
    print()
    
    rag = ReRankingRAG()
    
    if rag.collection.count() == 0:
        print("⚠️  No documents in database.")
        exit(1)
    
    question = "What does FIS use for data analytics?"
    
    print(f"Question: {question}\n")
    print("Stage 1: Vector search (retrieving 10 candidates)...")
    print("Stage 2: Re-ranking (selecting top 3)...\n")
    
    result = rag.query_with_reranking(
        question,
        initial_results=10,
        final_results=3
    )
    
    print(f"💡 Answer:\n{result['answer']}\n")
    print(f"📊 Re-ranking Statistics:")
    print(f"  Initial candidates: {result['reranking_stats']['initial_candidates']}")
    print(f"  Final results: {result['reranking_stats']['final_results']}")
    print()
    
    print("📈 Top 3 After Re-ranking:")
    for i, item in enumerate(result['reranking_stats']['reranking_impact'], 1):
        print(f"\n  {i}. {item['chunk_preview']}")
        print(f"     Vector:  {item['vector_score']}")
        print(f"     Rerank:  {item['rerank_score']}")
        print(f"     Combined: {item['combined']}")
    
    print("\n" + "-" * 70)
    print("\n📝 Enterprise Re-ranking Approaches:")
    print("  • Cross-encoders: BERT models for pairwise scoring")
    print("  • LLM-as-judge: GPT/Claude rates relevance 1-10")
    print("  • Learned-to-rank: ML models trained on click data")
    print("  • Hybrid: Combine multiple signals (recency, authority, etc.)")
