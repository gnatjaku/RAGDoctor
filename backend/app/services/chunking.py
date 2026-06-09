import logging
from typing import List

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Dzieli tekst na chunki o maksymalnej długości `chunk_size` znaków,
    z nakładaniem `chunk_overlap` znaków między sąsiednimi chunkami.

    Podział następuje po słowach – żadne słowo nie zostanie przecięte w połowie.

    Przykład (chunk_size=20, chunk_overlap=5):
        Tekst  : "Ala ma kota i psa oraz rybkę"
        chunk 1: "Ala ma kota i psa"       (≤20 znaków)
        chunk 2: "i psa oraz rybkę"        (zaczyna od ostatnich ~5 znaków chunk 1)
    """
    logger.debug(
        "chunk_text wywołany: chunk_size=%d, chunk_overlap=%d, długość tekstu=%d znaków",
        chunk_size, chunk_overlap, len(text),
    )

    # Normalizacja białych znaków
    text = " ".join(text.split())
    if not text:
        logger.warning("Przekazano pusty tekst – zwracam pustą listę chunków.")
        return []

    if chunk_overlap >= chunk_size:
        logger.error(
            "chunk_overlap (%d) musi być mniejszy niż chunk_size (%d)",
            chunk_overlap, chunk_size,
        )
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split(" ")
    logger.debug("Liczba słów po normalizacji: %d", len(words))
    chunks: list[str] = []
    start_word = 0

    while start_word < len(words):
        current_chars = 0
        end_word = start_word

        # Dodawaj słowa dopóki chunk nie przekroczy chunk_size znaków
        while end_word < len(words):
            word_len = len(words[end_word]) + (1 if end_word > start_word else 0)  # +1 za spację
            if current_chars + word_len > chunk_size and end_word > start_word:
                break
            current_chars += word_len
            end_word += 1

        chunk = " ".join(words[start_word:end_word])
        chunks.append(chunk)
        logger.debug("Chunk #%d (znaki=%d): %r", len(chunks), len(chunk), chunk)

        if end_word >= len(words):
            break

        # Cofnij się o overlap znaków – znajdź odpowiedni indeks słowa
        overlap_chars = 0
        next_start = end_word
        for i in range(end_word - 1, start_word, -1):
            overlap_chars += len(words[i]) + 1  # +1 za spację
            if overlap_chars >= chunk_overlap:
                next_start = i
                break

        # Zabezpieczenie przed nieskończoną pętlą
        if next_start >= end_word:
            next_start = end_word

        start_word = next_start

    logger.info("Podzielono tekst na %d chunków (chunk_size=%d, chunk_overlap=%d).", len(chunks), chunk_size, chunk_overlap)
    return chunks

