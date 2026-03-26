# 📄 Chunking tekstu – RAGDoctor

## Czym jest chunking?

**Chunking** to proces dzielenia długiego tekstu na mniejsze fragmenty zwane **chunkami**.  
Jest to kluczowy krok w systemach RAG (Retrieval-Augmented Generation), ponieważ:

- modele językowe (LLM) mają **ograniczone okno kontekstu**
- wyszukiwanie wektorowe działa najlepiej na **krótkich, spójnych fragmentach**
- mniejsze chunki = **dokładniejsze dopasowanie** do zapytania użytkownika

---

## Parametry

| Parametr | Domyślna wartość (`.env`) | Opis |
|---|---|---|
| `chunk_size` | `800` | Maksymalna liczba znaków w jednym chunku |
| `chunk_overlap` | `120` | Liczba znaków nakładających się między sąsiednimi chunkami |

---

## Jak działa `chunk_text()`?

Funkcja znajduje się w `app/services/chunking.py`.

### Algorytm krok po kroku

```
1. Normalizacja – usuwa wielokrotne spacje i znaki nowej linii
2. Podział tekstu na słowa (split po spacji)
3. Budowanie chunka – dodawaj słowa dopóki suma znaków ≤ chunk_size
4. Zapisz chunk
5. Cofnij się o ~chunk_overlap znaków (po granicy słowa) – to jest overlap
6. Wróć do kroku 3 aż do końca tekstu
```

> ✅ Podział następuje zawsze **po słowach** – żadne słowo nie zostanie przecięte w połowie.

---

## Wizualizacja

```
chunk_size    = 800  znaków
chunk_overlap = 120  znaków
krok          = 800 - 120 = 680 znaków

Tekst wejściowy (np. 2000 znaków):

 ┌─────────────────────────────────────────────────┐
 │                    chunk 1                      │  znaki [0 → 800]
 └────────────────────────────┬────────────────────┘
                              │← 120 znaków overlap
              ┌───────────────┴─────────────────────────────────┐
              │                    chunk 2                      │  znaki [680 → 1480]
              └───────────────────────────┬─────────────────────┘
                                          │← 120 znaków overlap
                          ┌──────────────┴──────────────────────────────────┐
                          │                    chunk 3                      │  znaki [1360 → 2000]
                          └─────────────────────────────────────────────────┘
```

---

## Po co jest overlap?

Bez overlapa zdanie może zostać **przecięte na granicy chunku**:

```
❌ Bez overlap:
   chunk 1: "...pacjent odczuwa silny"
   chunk 2: "ból w klatce piersiowej..."   ← LLM nie widzi pełnego kontekstu!

✅ Z overlap (120 znaków):
   chunk 1: "...pacjent odczuwa silny ból w klatce piersiowej..."
   chunk 2: "...silny ból w klatce piersiowej... i duszność..."  ← kontekst zachowany
```

---

## Przykład z małymi wartościami

```python
chunk_text("Ala ma kota i psa oraz rybkę", chunk_size=20, chunk_overlap=5)
```

```
Tekst : "Ala ma kota i psa oraz rybkę"
         |_________________|              chunk 1: "Ala ma kota i psa"  (17 znaków)
                       |_______________|  chunk 2: "i psa oraz rybkę"   (17 znaków)
                                                      ↑
                                               5 znaków overlap ("i psa")
```

Wynik:
```python
["Ala ma kota i psa", "i psa oraz rybkę"]
```

---

## Miejsce chunkingu w pipeline RAGDoctor

```
📄 PDF / TXT
     │
     ▼
[ ingest.py ]       – wczytaj plik, wyciągnij surowy tekst
     │
     ▼
[ chunking.py ]     – chunk_text(text, chunk_size=800, chunk_overlap=120)
     │
     │   chunk 1: "Pacjent lat 45 zgłosił..."       (800 znaków)
     │   chunk 2: "...zgłosił ból w klatce..."      (overlap 120 znaków)
     │   chunk 3: "...ból w klatce, duszność..."
     │
     ▼
[ embeddings.py ]   – embed_texts([chunk1, chunk2, ...])
     │               → OpenAI text-embedding-3-small
     │               → lista wektorów float[1536]
     ▼
[ MongoDB ]         – zapisz { text, embedding, chunk_id, source_name }
     │
     ▼
[ retrieve.py ]     – cosine_similarity(query_vector, all_embeddings)
     │               → top K najbardziej podobnych chunków
     ▼
[ answer.py ]       – GPT-4.1-mini + kontekst z chunków → odpowiedź
```

---

## Kod źródłowy

```python
# app/services/chunking.py

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = " ".join(text.split())        # 1. normalizacja
    words = text.split(" ")              # 2. podział na słowa

    while start_word < len(words):
        # 3. buduj chunk słowo po słowie (max chunk_size znaków)
        while end_word < len(words):
            if current_chars + word_len > chunk_size:
                break
            ...

        chunks.append(chunk)             # 4. zapisz chunk

        # 5. cofnij się o ~chunk_overlap znaków (po granicy słowa)
        for i in range(end_word - 1, start_word, -1):
            if overlap_chars >= chunk_overlap:
                next_start = i
                break

    return chunks
```

---

## Konfiguracja w `.env`

```dotenv
CHUNK_SIZE=800       # rozmiar chunku w znakach
CHUNK_OVERLAP=120    # nakładanie między chunkami w znakach
```

> 💡 **Wskazówka:** Dla tekstów medycznych zalecane wartości to `chunk_size=600–1000` i `chunk_overlap=100–150`.  
> Zbyt mały chunk = utrata kontekstu. Zbyt duży chunk = mniej precyzyjne wyszukiwanie.
