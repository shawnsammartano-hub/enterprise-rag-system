#!/usr/bin/env python3
"""
Chunking Strategy Analysis - Enterprise Best Practice

Why it matters:
- Too small: Loses context, fragmented answers
- Too large: Irrelevant info, slow embeddings
- Overlap: Preserves context across boundaries

Enterprise defaults:
- Databricks: 500-1000 tokens with 10-20% overlap
- Snowflake Cortex: 512 tokens
- OpenAI: 400-600 tokens with 100 token overlap
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.rag_pipeline import EnterpriseRAG

class ChunkingAnalyzer(EnterpriseRAG):
    """Analyze different chunking strategies"""
    
    def analyze_chunk_sizes(self, text: str, sizes: list = [200, 500, 1000]):
        """
        Compare different chunk sizes
        
        Trade-offs:
        - Small chunks (200): Precise, but miss context
        - Medium chunks (500): Balanced (enterprise default)
        - Large chunks (1000): More context, but noisy
        """
        print("📊 Chunking Strategy Analysis\n")
        print(f"Document length: {len(text.split())} words\n")
        
        results = {}
        for size in sizes:
            chunks = self.chunk_text(text, chunk_size=size, overlap=int(size * 0.1))
            
            avg_length = sum(len(c.split()) for c in chunks) / len(chunks)
            
            results[size] = {
                'num_chunks': len(chunks),
                'avg_words_per_chunk': round(avg_length, 1),
                'overlap': int(size * 0.1)
            }
            
            print(f"Chunk Size: {size} words")
            print(f"  Chunks created: {results[size]['num_chunks']}")
            print(f"  Avg words/chunk: {results[size]['avg_words_per_chunk']}")
            print(f"  Overlap: {results[size]['overlap']} words")
            print()
        
        return results
    
    def demonstrate_overlap_impact(self, text: str):
        """
        Show why overlap matters
        
        Without overlap: Sentences split at boundaries lose meaning
        With overlap: Context preserved across chunks
        """
        print("🔗 Overlap Impact Demonstration\n")
        
        # No overlap
        no_overlap = self.chunk_text(text, chunk_size=100, overlap=0)
        print(f"No Overlap:")
        print(f"  Chunks: {len(no_overlap)}")
        print(f"  Last words of chunk 1: ...{' '.join(no_overlap[0].split()[-5:])}")
        print(f"  First words of chunk 2: {' '.join(no_overlap[1].split()[:5])}...")
        print()
        
        # With overlap
        with_overlap = self.chunk_text(text, chunk_size=100, overlap=20)
        print(f"20% Overlap (20 words):")
        print(f"  Chunks: {len(with_overlap)}")
        print(f"  Last words of chunk 1: ...{' '.join(with_overlap[0].split()[-5:])}")
        print(f"  First words of chunk 2: {' '.join(with_overlap[1].split()[:5])}...")
        print()
        
        print("✓ Overlap ensures important context isn't lost at boundaries")

# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("CHUNKING STRATEGY ANALYSIS - Enterprise RAG Best Practices")
    print("=" * 70)
    print()
    
    # Load sample document
    doc_path = Path(__file__).parent.parent / "documents" / "fintech_overview.txt"
    
    if not doc_path.exists():
        print(f"⚠️  Document not found: {doc_path}")
        print("Upload documents first via the web UI.")
        exit(1)
    
    with open(doc_path, 'r') as f:
        text = f.read()
    
    analyzer = ChunkingAnalyzer()
    
    # Analyze different chunk sizes
    results = analyzer.analyze_chunk_sizes(text, sizes=[200, 500, 1000])
    
    print("-" * 70)
    print()
    
    # Demonstrate overlap
    analyzer.demonstrate_overlap_impact(text)
    
    print("-" * 70)
    print("\n📝 Enterprise Recommendations:")
    print("  • Start with 500 words, 10% overlap")
    print("  • Technical docs: 300-400 words (dense info)")
    print("  • Narratives/reports: 700-1000 words (more context)")
    print("  • Always measure retrieval quality with your data")
