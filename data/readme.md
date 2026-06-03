# RAG

## Napotkane problemy

---

### Problem 1: RAG nie odpowiada z kontekstu mimo że chunk z odpowiedzią jest w bazie

**Data:** 2026-04-12

---

#### 🔍 Diagnoza

Zapytanie: *"Podaj wirusy wywołujące grypę sezonową?"*

RAG zwracał chunki o **HIV** zamiast o **grypie**, mimo że w bazie znajdował się chunk `treatment-001:1` z dokładną odpowiedzią (`H1N1`, `H3N2`).

Uruchomiony skrypt `data/diagnose_rag.py` oraz `data/diag2.py` wykazał:

**TOP cosine similarity (przed naprawą):**

| Score | chunk_id | Treść |
|---|---|---|
| 0.7552 | `treatment-001:1` ✅ | *"Grypę sezonową wywołują wirusy H1N1 i H3N2..."* |
| 0.7543 | `medonet HIV` ❌ | *"...ryzyko zarażenia wirusem HIV..."* |
| 0.7427 | `medonet HIV` ❌ | *"...faza wirusa HIV..."* |
| 0.7297 | `treatment-001:0` ✅ | *"Grypa jest ostrą chorobą zakaźną..."* |

Kluczowe liczby z diagnostyki:

```
Wymiar embeddingu w bazie:      768
Wymiar embeddingu nomic live:   768  ← OK, zgodne
Similarity pytanie vs chunk grypa:  0.6323
Similarity flu chunk vs HIV chunk:  0.8286  ← MODEL UWAŻA JE ZA PRAWIE IDENTYCZNE!
```

**Indeksy MongoDB (przed naprawą):**
```
_id_              : {_id: 1}
chunk_id_unique   : {chunk_id: 1}
embedding_plain   : {embedding: 1}   ← zwykły indeks, nie wektorowy!
```
Retrieval działał w trybie brute-force cosine (pełny skan kolekcji) — poprawnie technicznie, ale jakość wyników zła.

---

#### 🧨 Przyczyna

**`nomic-embed-text` jest trenowany głównie na tekstach angielskich.**
Dla polskich tekstów medycznych wszystkie chunki zawierające słowo *"wirus"* otrzymują bardzo podobne embeddingi — model nie potrafi semantycznie odróżnić "wirusa grypy" od "wirusa HIV" w języku polskim.

Różnica cosine similarity między właściwym chunkiem (grypa) a błędnym (HIV) wynosiła zaledwie **0.001** (`0.7552` vs `0.7543`). Przy `top_k=3` kolejność w bazie decydowała o tym, który chunk wygrywał.

---

#### ✅ Rozwiązanie

Dodano **hybrydowy re-ranking** w `app/services/retrieve.py`:

```python
# Wynik hybrydowy: 70% semantyczny (cosine) + 30% keyword
hybrid = 0.70 * vec_score + 0.30 * kw_score
```

Funkcja `_keyword_score()` liczy pokrycie słów kluczowych z pytania w tekście chunka z uproszczonym stemmingiem (pierwsze 5 liter), pomijając polskie stop-słowa.

**TOP cosine similarity (po naprawie):**

| Hybrid score | Vec score | KW score | chunk_id |
|---|---|---|---|
| **0.6388** | 0.6125 | 0.70 | `treatment-001:1` ✅ |
| **0.6076** | 0.5679 | 0.70 | `treatment-001:0` ✅ |
| **0.5750** | 0.5642 | 0.60 | `medonet-701135cd:3` ✅ |

Chunki o HIV zostały zepchnięte poniżej, bo zawierają słowo "wirus" ale nie "grypa/sezonowa".

**Odpowiedź RAG po naprawie:**
> *"Grypę sezonową wywołują najczęściej wirusy podtypów H1N1 i H3N2 (w niektórych sezonach H1N2)"* ✅

---

#### 📁 Pliki diagnostyczne

| Plik | Opis |
|---|---|
| `data/diagnose_rag.py` | Skrypt sprawdzający zawartość bazy, wymiary embeddingów, indeksy |
| `data/diag2.py` | Skrypt obliczający pełny ranking cosine similarity dla wszystkich chunków |
| `data/diagnose_out.txt` | Wynik diagnostyki — 44 chunki, 4 źródła, wymiar 768 |
| `data/diag2_out.txt` | Wynik rankingu — similarity grypa vs HIV |
| `app/services/retrieve.py` | **Naprawiony plik** — hybrydowy retrieval |

---

#### 💡 Długoterminowe rekomendacje

1. **Zmień model embeddingów na wielojęzyczny**, np.:
   ```bash
   ollama pull mxbai-embed-large   # 1024 dim, lepsza jakość
   ```
   lub użyj `multilingual-e5-large` (wymaga re-ingestion wszystkich chunków).

2. **Re-ingest po zmianie modelu** — embeddingi w bazie muszą być z tego samego modelu co embeddingi pytania.

3. **Zwiększ `top_k`** — np. `RAG_TOP_K=8` daje więcej kandydatów i zmniejsza ryzyko pominięcia trafnego chunka.
