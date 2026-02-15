 #!/usr/bin/env python3
"""
Embedding Model Comparison - Enterprise Trade-offs

Key metrics:
- Dimensions: Higher = more nuanced, but slower & more storage
- Speed: Tokens/second for indexing & search
- Quality: How well it captures semantic meaning
- Cost: Compute & storage requirements

Common enterprise models:
- OpenAI text-embedding-3-small: 1536 dims, fast, $$$
- OpenAI text-embedding-3-large: 3072 dims, slow, $$$$
- nomic-embed-text: 768 dims, free, local
- bge-large: 1024 dims, good quality, free
- E5-large: 1024 dims, good for domain-specific

Databricks: Uses instructor-xl (768 dims) by default
Snowflake: Multiple options (e5, bge, etc.)
"""

import sys
from pathlib import Path
import time
sys.path.append(str(Path(__file__).parent.parent))

from scripts.rag_pipeline import EnterpriseRAG
import requests

class EmbeddingComparison:
    """Compare different embedding approaches"""
    
    def __init__(self):
        self.ollama_base = "http://localhost:11434"
    
    def benchmark_embedding_speed(self, model: str, text: str, runs: int = 3):
        """
        Measure embedding generation speed
        
        Important for:
        - Bulk ingestion (indexing millions of docs)
        - Real-time search (query latency)
        """
        times = []
        
        for _ in range(runs):
            start = time.time()
            
            response = requests.post(
                f"{self.ollama_base}/api/embeddings",
                json={"model": model, "prompt": text}
            )
            
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        
        # Get embedding dimensions
        embedding = response.json()["embedding"]
        dimensions = len(embedding)
        
        return {
            "model": model,
            "avg_time_ms": round(avg_time * 1000, 2),
            "dimensions": dimensions,
            "tokens": len(text.split()),
            "tokens_per_sec": round(len(text.split()) / avg_time, 1)
        }
    
    def compare_models(self, text: str):
        """
        Compare available embedding models
        
        Trade-offs:
        - Small models: Fast, less nuanced
        - Large models: Slow, more nuanced
        - Local vs API: Privacy vs convenience
        """
        print("🔬 Embedding Model Comparison\n")
        print(f"Sample text: '{text[:80]}...'\n")
        print(f"Text length: {len(text.split())} words\n")
        
        # Models to compare (only ones available in Ollama)
        models = ["nomic-embed-text"]
        
        # Check what's available
        try:
            response = requests.get(f"{self.ollama_base}/api/tags")
            available = [m["name"] for m in response.json()["models"]]
            print(f"Available models: {', '.join(available)}\n")
        except:
            pass
        
        results = []
        for model in models:
            try:
                print(f"Testing {model}...", end=" ")
                result = self.benchmark_embedding_speed(model, text)
                results.append(result)
                print("✓")
            except Exception as e:
                print(f"✗ ({str(e)})")
        
        return results
    
    def analyze_dimension_impact(self):
        """
        Explain dimension trade-offs
        """
        print("\n" + "=" * 70)
        print("EMBEDDING DIMENSIONS - Trade-off Analysis")
        print("=" * 70)
        print()
        
        comparisons = [
            {
                "model": "Small (384 dims)",
                "examples": "all-MiniLM-L6",
                "speed": "⚡⚡⚡ Very Fast",
                "quality": "⭐⭐ Good",
                "storage": "💾 4x smaller",
                "use_case": "High-volume, speed-critical"
            },
            {
                "model": "Medium (768 dims)",
                "examples": "nomic-embed-text, instructor-xl",
                "speed": "⚡⚡ Fast",
                "quality": "⭐⭐⭐ Very Good",
                "storage": "💾 2x smaller",
                "use_case": "Enterprise default (Databricks)"
            },
            {
                "model": "Large (1024-1536 dims)",
                "examples": "bge-large, OpenAI-small",
                "speed": "⚡ Moderate",
                "quality": "⭐⭐⭐⭐ Excellent",
                "storage": "💾 Standard",
                "use_case": "Quality-critical applications"
            },
            {
                "model": "XL (3072+ dims)",
                "examples": "OpenAI-large",
                "speed": "🐌 Slow",
                "quality": "⭐⭐⭐⭐⭐ Best",
                "storage": "💾 2x larger",
                "use_case": "Research, maximum accuracy"
            }
        ]
        
        for comp in comparisons:
            print(f"📊 {comp['model']}")
            print(f"   Examples:  {comp['examples']}")
            print(f"   Speed:     {comp['speed']}")
            print(f"   Quality:   {comp['quality']}")
            print(f"   Storage:   {comp['storage']}")
            print(f"   Use Case:  {comp['use_case']}")
            print()

# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("EMBEDDING MODEL ANALYSIS - Enterprise Decision Making")
    print("=" * 70)
    print()
    
    comparator = EmbeddingComparison()
    
    # Sample text
    sample_text = """Fintech Company is a Fortune 500 company providing financial 
    technology solutions to firms worldwide. 
    The company leverages Databricks for data analytics and Snowflake for 
    data warehousing."""
    
    # Benchmark available models
    results = comparator.compare_models(sample_text)
    
    if results:
        print("\n📈 Performance Results:\n")
        for r in results:
            print(f"Model: {r['model']}")
            print(f"  Dimensions: {r['dimensions']}")
            print(f"  Avg time: {r['avg_time_ms']}ms")
            print(f"  Speed: {r['tokens_per_sec']} tokens/sec")
            print()
    
    # Dimension analysis
    comparator.analyze_dimension_impact()
    
    print("-" * 70)
    print("\n💡 Enterprise Recommendations:")
    print("  • Start with 768 dims (nomic/instructor)")
    print("  • Benchmark with your actual queries")
    print("  • Consider: Is 2x slower worth 5% better accuracy?")
    print("  • Local models: Privacy + cost savings")
    print("  • API models: Better quality + convenience")
    print("\n📊 Storage Impact (1M documents):")
    print("  • 384 dims = ~1.5 GB")
    print("  • 768 dims = ~3 GB")
    print("  • 1536 dims = ~6 GB")
    print("  • 3072 dims = ~12 GB")
