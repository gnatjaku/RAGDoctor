#!/usr/bin/env python3
"""Skrypt diagnostyczny RAG - sprawdza embeddingi i chunki w MongoDB"""
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from pymongo import MongoClient

sys.path.insert(0, '/home/jakub/PycharmProjects/RAGDoctor')

load_dotenv(Path(__file__).parent.parent / ".env")

MONGO_URI = os.getenv("MONGODB_URI") or (
    f"mongodb://{os.getenv('MONGO_INITDB_ROOT_USERNAME', 'admin')}:"
    f"{os.getenv('MONGO_INITDB_ROOT_PASSWORD', '')}@localhost:27017/?authSource=admin"
)
DB_NAME = os.getenv("MONGODB_DB", "rag_db")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "documents")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
QUERY = os.getenv("DIAGNOSE_RAG_QUERY", "wirusy wywołujące grypę sezonową")

c = MongoClient(MONGO_URI)
col = c[DB_NAME][COLLECTION_NAME]

print("=" * 60)
print("1. ZAWARTOŚĆ BAZY MONGODB")
print("=" * 60)
total = col.count_documents({})
print(f"Łączna liczba chunków: {total}")

sources = col.distinct('source_name')
print(f"\nŹródła ({len(sources)}):")
for s in sources:
    cnt = col.count_documents({'source_name': s})
    print(f"  [{cnt:3d}] {s}")

print("\n" + "=" * 60)
print("2. CHUNKI O GRYPIE/INFLUENZA")
print("=" * 60)
flu = list(col.find(
    {'$or': [
        {'text': {'$regex': 'gryp', '$options': 'i'}},
        {'text': {'$regex': 'influenza', '$options': 'i'}},
        {'text': {'$regex': 'Influenza', '$options': 'i'}},
    ]},
    {'chunk_id': 1, 'source_name': 1, 'text': 1, '_id': 0}
))
print(f"Znaleziono {len(flu)} chunków o grypie/influenza")
for i, ch in enumerate(flu[:5]):
    print(f"\n  [{i+1}] chunk_id: {ch.get('chunk_id')}")
    print(f"       source: {ch.get('source_name')}")
    print(f"       text:   {ch.get('text','')[:200]}")

print("\n" + "=" * 60)
print("3. SPRAWDZENIE EMBEDDINGÓW W BAZIE")
print("=" * 60)
sample = col.find_one({}, {'embedding': 1, 'source_name': 1, '_id': 0})
if sample and 'embedding' in sample:
    emb = sample['embedding']
    print(f"Wymiar embeddingu w bazie (chunk z '{sample.get('source_name','?')}'): {len(emb)}")
    print(f"Typ wartości: {type(emb[0])}")
    print(f"Pierwsze 3 wartości: {emb[:3]}")
else:
    print("BRAK pola 'embedding' w dokumentach! Chunki nie mają embeddingów.")

print("\n" + "=" * 60)
print(f"4. TEST OLLAMA {EMBED_MODEL}")
print("=" * 60)
try:
    r = requests.post(f'{OLLAMA_URL}/api/embed',
        json={'model': EMBED_MODEL, 'input': QUERY}, timeout=30)
    r.raise_for_status()
    data = r.json()
    emb_q = data.get('embeddings', [[]])[0]
    print(f"Pytanie: '{QUERY}'")
    print(f"Wymiar embeddingu: {len(emb_q)}")
    print(f"Pierwsze 3 wartości: {emb_q[:3]}")
except Exception as e:
    print(f"BŁĄD: {e}")

print("\n" + "=" * 60)
print("5. INDEKSY MONGODB")
print("=" * 60)
indexes = list(col.list_indexes())
for idx in indexes:
    print(f"  {idx.get('name')}: {idx.get('key')}")

print("\n" + "=" * 60)
print("6. COSINE SIMILARITY - pytanie vs chunk grypa")
print("=" * 60)
if flu:
    # pobierz embedding chunka o grypie
    flu_with_emb = col.find_one(
        {'chunk_id': flu[0]['chunk_id']}, {'embedding': 1, '_id': 0}
    )
    if flu_with_emb and 'embedding' in flu_with_emb:
        import math
        e1 = emb_q
        e2 = flu_with_emb['embedding']
        if len(e1) == len(e2):
            dot = sum(a * b for a, b in zip(e1, e2))
            n1 = math.sqrt(sum(a * a for a in e1))
            n2 = math.sqrt(sum(b * b for b in e2))
            cos = dot / (n1 * n2) if n1 and n2 else 0
            print(f"Cosine similarity pytanie ↔ chunk_grypa: {cos:.4f}")
        else:
            print(f"RÓŻNE WYMIARY! pytanie={len(e1)}, chunk={len(e2)} — TO JEST PROBLEM!")
    else:
        print("Chunk grypy nie ma embeddingu w bazie")

print("\nDone.")
