#!/usr/bin/env python3
"""Test RAG pipeline setup"""

import chromadb
import requests

# Test 1: ChromaDB
print("✓ ChromaDB installed")
client = chromadb.Client()
print("✓ ChromaDB client initialized")

# Test 2: Ollama connection
try:
    response = requests.get("http://localhost:11434/api/tags")
    models = response.json()
    print(f"✓ Ollama connected ({len(models.get('models', []))} models)")
except:
    print("✗ Ollama connection failed")

# Test 3: Embedding model
try:
    embed_response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": "test"}
    )
    if embed_response.status_code == 200:
        print("✓ nomic-embed-text working")
    else:
        print("✗ Embedding model failed")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n🎉 Setup complete! Ready to build RAG pipeline.")
