# RAGDoctor

**RAGDoctor** to edukacyjny prototyp RAG dla dokumentów medycznych. Projekt pokazuje, jak zbudować asystenta pytaniowego nad przygotowanym kontekstem medycznym: od ingestu dokumentów, przez chunking i embeddingi, po retrieval, re-ranking i odpowiedź generowaną przez lokalny model LLM.

Projekt nie jest narzędziem diagnostycznym, systemem wspierania decyzji klinicznych ani zamiennikiem konsultacji medycznej. Jego celem jest demonstracja architektury AI/RAG i praktycznych problemów związanych z jakością retrievalu.

## Dla kogo

README jest skierowany do prowadzącego zajęcia lub komisji oceniającej projekt techniczny, szczególnie od strony AI. Opisuje zarówno główną koncepcję RAG, jak i ważniejsze elementy techniczne: lokalne modele w Ollamie, MongoDB jako magazyn chunków i embeddingów, FastAPI, Web GUI oraz eksport PDF.

## Najważniejsze założenia

- Odpowiedzi są generowane na podstawie zaindeksowanych dokumentów, a nie swobodnej wiedzy modelu.
- Domyślny tryb AI korzysta lokalnie z Ollamy: model czatu `doctor` oraz embeddingi `nomic-embed-text`.
- MongoDB przechowuje chunki dokumentów, embeddingi i metadane źródeł.
- Retrieval łączy cosine similarity z hybrydowym re-rankingiem leksykalnym.
- API udostępnia ingest dokumentów, zadawanie pytań i eksport odpowiedzi do PDF.
- Web GUI jest prostym demonstratorem do zadawania pytań i podglądu użytych chunków.

## Pipeline RAG

```text
Dokument medyczny
        |
        v
Chunking tekstu z overlapem
        |
        v
Embeddingi chunków
        |
        v
MongoDB: chunki + embeddingi + metadane
        |
        v
Pytanie użytkownika
        |
        v
Embedding pytania
        |
        v
Hybrid retrieval + re-ranking
        |
        v
LLM: Ollama albo OpenAI
        |
        v
Odpowiedź + cytowane chunki
```

Najpierw dokument jest dzielony na mniejsze fragmenty, czyli chunki. Każdy chunk otrzymuje embedding i jest zapisywany w MongoDB razem z metadanymi źródła. Gdy użytkownik zada pytanie, system tworzy embedding pytania, wyszukuje podobne chunki, poprawia ranking dodatkowymi regułami leksykalnymi, a następnie przekazuje wybrane fragmenty do LLM jako kontekst odpowiedzi.

## Architektura

```text
Użytkownik / Web GUI
        |
        v
Node.js + Express proxy
        |
        v
FastAPI
  |     |      |
  |     |      +--> PDF export
  |     +---------> Ollama / OpenAI
  +---------------> MongoDB
```

Najważniejsze katalogi:

```text
app/
  main.py              # FastAPI i endpointy
  services/
    chunking.py        # dzielenie dokumentów na chunki
    embeddings.py      # embeddingi OpenAI/Ollama
    retrieve.py        # retrieval i hybrydowy re-ranking
    answer.py          # generowanie odpowiedzi z kontekstu
    pdf_export.py      # eksport PDF
data/
  scrape_and_ingest.py # pobieranie i ingest przykładowych artykułów
  diagnose_rag.py      # diagnostyka chunków, embeddingów i similarity
models/
  Modelfile            # lokalna konfiguracja modelu Ollama
webgui/
  server.js            # prosty serwer Express i proxy do API
```

## Technologie

- **Python 3.12**: backend aplikacji.
- **FastAPI**: API dla ingestu, pytań i eksportu PDF.
- **MongoDB**: magazyn chunków, embeddingów i metadanych.
- **PyMongo**: komunikacja z MongoDB.
- **PyTorch**: lokalne liczenie cosine similarity na CPU albo GPU.
- **Ollama**: lokalny backend AI dla LLM i embeddingów.
- **OpenAI API**: alternatywny backend dla modelu czatu i embeddingów.
- **Node.js + Express**: prosty Web GUI i proxy do FastAPI.
- **ReportLab**: generowanie PDF z odpowiedzią.
- **Docker Compose**: uruchomienie MongoDB, API i narzędzi pomocniczych.

## Lokalny backend AI

Domyślnie projekt korzysta z lokalnej Ollamy. Zapytania do LLM i embeddingów nie muszą opuszczać lokalnego środowiska, o ile MongoDB, API i Ollama działają lokalnie.

Model czatu `doctor` jest lokalną konfiguracją Ollamy opartą o `gemma3:12b`. Może zostać utworzony z pliku `models/Modelfile`. Nie jest to osobny model medyczny trenowany od podstaw. Prompt systemowy wymusza odpowiadanie na podstawie dostarczonego kontekstu, unikanie zmyślania oraz zachęcanie do konsultacji medycznej w razie wątpliwości.

Przygotowanie modeli:

```bash
ollama pull gemma3:12b
ollama pull nomic-embed-text
ollama create doctor -f models/Modelfile
ollama serve
```

Po zmianie modelu embeddingów trzeba ponownie zaindeksować dokumenty, ponieważ embedding pytania i embeddingi chunków muszą pochodzić z tego samego modelu.

## Retrieval i re-ranking

Projekt pokazuje praktyczny problem RAG: samo podobieństwo wektorowe nie zawsze wystarcza, szczególnie dla polskich tekstów medycznych i lokalnych modeli embeddingów. W udokumentowanym przypadku `nomic-embed-text` zwracał bardzo zbliżone wyniki dla chunków o grypie i HIV, bo teksty miały podobne słowa medyczne.

Dlatego `app/services/retrieve.py` łączy kilka sygnałów:

- cosine similarity embeddingów,
- dopasowanie słów kluczowych z pytania,
- dopasowanie tytułu i nazwy źródła,
- proste normalizowanie polskich form wyrazów,
- premie za zgodność tematu pytania i intencji.

Taki hybrydowy retrieval poprawia trafność wyboru chunków, gdy embeddingi są zbyt mało rozróżniające.

## MongoDB jako magazyn chunków i embeddingów

MongoDB przechowuje dokumenty w postaci:

- `source_id`: stabilny identyfikator dokumentu,
- `source_name`: nazwa pliku lub źródła,
- `chunk_id`: identyfikator chunka,
- `text`: treść chunka,
- `embedding`: wektor embeddingu,
- `metadata`: dodatkowe metadane, np. URL lub kategoria.

W lokalnym trybie aplikacja pobiera embeddingi z MongoDB i liczy podobieństwo kosinusowe po stronie aplikacji. Kod zawiera też przygotowanie indeksów MongoDB, w tym obsługę Atlas Vector Search, ale lokalny fallback nie jest pełnoprawnym wyszukiwaniem wektorowym.

## API

Backend FastAPI udostępnia najważniejsze endpointy:

- `GET /health`: sprawdzenie statusu API.
- `POST /ingest`: dodanie dokumentu do magazynu chunków i embeddingów.
- `POST /ask`: zadanie pytania i wygenerowanie odpowiedzi z cytowanymi chunkami.
- `POST /answers/pdf`: eksport odpowiedzi do pliku PDF.

Przykład ingestu:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "treatment-001",
    "source_name": "treatment-flu.txt",
    "text": "Grypa jest ostrą chorobą zakaźną dróg oddechowych...",
    "metadata": {"category": "treatment"}
  }'
```

Przykład pytania:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Podaj wirusy wywołujące grypę sezonową?",
    "top_k": 3
  }'
```

Odpowiedź zawiera tekst odpowiedzi oraz listę chunków, które zostały użyte jako kontekst.

## Web GUI

Web GUI jest lekkim interfejsem demonstracyjnym opartym o Node.js i Express. Udostępnia formularz zadawania pytań, pokazuje odpowiedź modelu, listę użytych chunków wraz z wynikami dopasowania oraz pozwala pobrać odpowiedź jako PDF.

Uruchomienie:

```bash
cd webgui
npm install
npm start
```

Domyślnie interfejs działa pod adresem `http://localhost:3000` i komunikuje się z FastAPI przez `RAG_API_URL`, domyślnie `http://localhost:8000`.

## Uruchomienie lokalne

1. Przygotuj plik `.env`:

```bash
cp .env.example .env
```

2. Ustaw bezpieczne hasło MongoDB w `.env`.

3. Przygotuj Ollamę i modele:

```bash
ollama pull gemma3:12b
ollama pull nomic-embed-text
ollama create doctor -f models/Modelfile
ollama serve
```

4. Uruchom MongoDB, indeksy i API:

```bash
docker compose -f docker-compose-rag.yml up -d mongodb
docker compose -f docker-compose-rag.yml run --rm init-indexes
docker compose -f docker-compose-rag.yml up -d rag-api
```

5. Sprawdź status API:

```bash
curl http://localhost:8000/health
```

6. Zaindeksuj przykładowy dokument:

```bash
bash data/ingest.sh
```

## Przykładowe pytania

Przykłady z katalogu `data/`:

- „Podaj wirusy wywołujące grypę sezonową?”
- „W jaki sposób można zarazić się wirusem HIV?”
- „Jak przebiega typowy zawał?”

Odpowiedzi zależą od tego, jakie dokumenty zostały wcześniej zaindeksowane. Jeśli odpowiedź nie wynika z kontekstu, model powinien zakomunikować brak wystarczających informacji.

## Źródła danych demonstracyjnych

Projekt może indeksować ręcznie dostarczone teksty oraz przykładowe artykuły medyczne pobrane skryptem demonstracyjnym `data/scrape_and_ingest.py`. Dokumenty są traktowane jako materiał edukacyjny do testowania pipeline'u RAG, a nie jako zweryfikowana baza wiedzy medycznej.

## Diagnostyka jakości RAG

Projekt zawiera skrypty diagnostyczne pozwalające sprawdzić liczbę chunków w MongoDB, źródła, wymiar embeddingów, indeksy oraz wyniki similarity dla przykładowych pytań. Ułatwia to analizę typowych problemów RAG, np. gdy poprawny chunk istnieje w bazie, ale nie trafia do top K.

Najważniejszy skrypt diagnostyczny:

```bash
python data/diagnose_rag.py
```

Szczegóły jednego z takich problemów opisuje `data/readme.md`.

## Czego projekt uczy

- Jak zbudować pipeline RAG od ingestu po odpowiedź.
- Czym różni się odpowiedź z wiedzy modelu od odpowiedzi opartej o kontekst.
- Dlaczego chunking i overlap mają znaczenie dla jakości odpowiedzi.
- Jak użyć lokalnych modeli przez Ollamę.
- Jakie ograniczenia mogą mieć lokalne embeddingi dla języka polskiego.
- Dlaczego retrieval często wymaga re-rankingu, a nie tylko similarity embeddingów.
- Jak przechowywać chunki, embeddingi i metadane w MongoDB.
- Jak udostępnić prototyp AI przez API i prosty interfejs webowy.

## Ograniczenia

RAGDoctor nie diagnozuje, nie zaleca leczenia i nie zastępuje konsultacji medycznej. Odpowiedzi są generowane wyłącznie na podstawie zaindeksowanego kontekstu, który może być niepełny, nieaktualny lub błędnie dobrany przez retrieval.

Lokalny tryb Ollama ogranicza potrzebę wysyłania zapytań do zewnętrznego API, ale nie oznacza automatycznie zgodności z wymaganiami ochrony danych medycznych. Projekt nie implementuje produkcyjnych mechanizmów bezpieczeństwa, audytu, kontroli dostępu ani zarządzania danymi wrażliwymi.

## Możliwe kierunki rozwoju

- Użycie lepszego wielojęzycznego modelu embeddingów.
- Porównanie Ollama vs OpenAI na tych samych pytaniach i dokumentach.
- Pełne użycie Atlas Vector Search albo innego silnika vector search.
- Automatyczne testy jakości retrieval dla zestawu pytań kontrolnych.
- Poprawa promptu systemowego i bezpieczniejszego języka medycznego.
- Rozbudowa Web GUI o ingest dokumentów z poziomu przeglądarki.
- Transkrypcja dźwięku, np. zadawanie pytań głosem i zamiana audio na tekst.
- Tryb agentowy z narzędziami MCP, np. wysłanie wygenerowanej odpowiedzi e-mailem albo zapisanie jej w zewnętrznym systemie.

Akcje agentowe powinny być oddzielone od samego mechanizmu RAG: RAG odpowiada na podstawie dokumentów, a agent wykonuje dodatkową akcję na wyniku.

## Dodatkowa dokumentacja

- `README_CHUNKING.md`: szczegóły chunkowania i overlapa.
- `README_DOCKER_MONGO.md`: uruchomienie MongoDB w Dockerze.
- `data/readme.md`: diagnostyka problemów retrieval i embeddingów.

## Informacja o sposobie tworzenia projektu

Aplikacja powstała jako przykład **Vibe Codingu**, czyli współpracy z narzędziami AI przy szybkim tworzeniu i iterowaniu kodu. Sam README został przygotowany przy użyciu skilla Matta Pococka, który reprezentuje bardziej kontrolowane podejście do pracy z AI: najpierw doprecyzowanie założeń, terminologii i decyzji projektowych, a dopiero potem generowanie dokumentacji.
