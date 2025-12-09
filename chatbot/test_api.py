#!/usr/bin/env python
"""Test script for RAG API."""
import requests
import json

BASE_URL = "http://localhost:8001"

print("=" * 60)
print("Testing RAG API")
print("=" * 60)

# Test health
print("\n1. Testing /health endpoint:")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test root
print("\n2. Testing / endpoint:")
try:
    response = requests.get(f"{BASE_URL}/", timeout=5)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test query
print("\n3. Testing /query endpoint:")
try:
    payload = {
        "query": "¿Qué es girar?",
        "topk": 3,
        "model": "gpt-4o-mini"
    }
    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        timeout=30
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   Query: {result['query']}")
        print(f"   Answer: {result['answer'][:200]}...")
        print(f"   Model: {result['model']}")
        print(f"   Topk: {result['topk']}")
        print(f"   ✓ SUCCESS!")
    else:
        print(f"   Error: {response.json()}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
