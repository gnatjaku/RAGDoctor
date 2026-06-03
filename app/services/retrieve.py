import logging
import re
import unicodedata

import torch
import torch.nn.functional as F

from app.db import collection
from app.config import settings
from app.services.embeddings import embed_texts

logger = logging.getLogger(__name__)

# Urządzenie obliczeniowe — GPU jeśli dostępne, w przeciwnym razie CPU
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("retrieve: urządzenie do obliczeń similarity: %s", _DEVICE)

# Polskie stop-słowa – nie wnoszą wartości do keyword boost
_STOP_WORDS = {
    "i", "w", "z", "na", "do", "się", "że", "nie", "to", "jest",
    "są", "jak", "czy", "co", "po", "przez", "dla", "przy", "od",
    "ale", "lub", "oraz", "też", "już", "jeszcze", "być", "który",
    "która", "które", "tego", "tej", "ten", "ta", "te", "ich", "jej",
    "jego", "go", "im", "się", "by", "więc", "a", "o", "u", "może",
}

_INTENT_TOKENS = {
    "leczyc", "leczenie", "leczyc", "objaw", "objawy", "przyczyna", "przyczyny",
    "diagnoza", "diagnostyka", "rozpoznac", "rozpoznanie", "zapobiegac",
    "profilaktyka", "terapia", "terapie", "badanie", "badania", "zakazenie",
    "zakazenia", "zarazic", "zarazenie",
}

_POLISH_SUFFIXES = (
    "owego", "owej", "owymi", "owym", "owie", "aniu", "enie", "ania", "enia",
    "iach", "owie", "owego", "owej", "ami", "ach", "ego", "owa", "owe", "owi",
    "cie", "ciu", "sci", "ści", "cja", "cje", "cji", "cją", "cjach",
    "nia", "nie", "niu", "ami", "ach", "ego", "emu", "owi", "owa", "owe",
    "ych", "ymi", "ach", "ami", "cie", "ciu", "ow", "om", "em", "ie", "ia",
    "iu", "ię", "ą", "ę", "a", "y", "i", "u", "e",
)


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_match_text(text: str) -> str:
    return _strip_accents(text.lower())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", _normalize_match_text(text))


def _stem_token(token: str) -> str:
    token = _normalize_match_text(token)
    canonical_prefixes = {
        "lecz": "lecz",
        "objaw": "objaw",
        "diagno": "diagno",
        "rozpozn": "rozpozn",
        "przyczyn": "przyczyn",
        "profil": "profil",
        "terap": "terap",
        "badan": "badan",
        "zakaz": "zakaz",
        "zaraz": "zaraz",
    }
    for prefix, canonical in canonical_prefixes.items():
        if token.startswith(prefix):
            return canonical
    for suffix in _POLISH_SUFFIXES:
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            stripped = token[: -len(suffix)]
            for prefix, canonical in canonical_prefixes.items():
                if stripped.startswith(prefix):
                    return canonical
            return stripped
    return token


def _split_query_tokens(query: str) -> tuple[list[str], list[str]]:
    tokens = [t for t in _tokenize(query) if t not in _STOP_WORDS]
    if not tokens:
        return [], []

    focus_tokens = [t for t in tokens if t not in _INTENT_TOKENS]
    intent_tokens = [t for t in tokens if t in _INTENT_TOKENS]

    if not focus_tokens:
        focus_tokens = tokens

    return focus_tokens, intent_tokens


def _token_match_score(tokens: list[str], text: str) -> float:
    if not tokens:
        return 0.0

    text_normalized = _normalize_match_text(text)
    text_tokens = set(_tokenize(text_normalized))
    text_stems = {_stem_token(t) for t in text_tokens}

    hits = 0
    for token in tokens:
        stem = _stem_token(token)
        if token in text_normalized:
            hits += 1
        elif stem in text_stems:
            hits += 0.8
        elif len(stem) >= 5 and stem[:5] in {s[:5] for s in text_stems if len(s) >= 5}:
            hits += 0.5
    return hits / len(tokens)


def _phrase_focus_score(focus_tokens: list[str], *fields: str) -> float:
    if not focus_tokens:
        return 0.0
    combined = " ".join(_normalize_match_text(field) for field in fields if field)
    stems_in_text = {_stem_token(token) for token in _tokenize(combined)}
    focus_stems = [_stem_token(token) for token in focus_tokens]
    matched = sum(1 for stem in focus_stems if stem in stems_in_text)
    return matched / len(focus_stems)


def _rerank_components(question: str, doc: dict) -> dict[str, float]:
    text = doc.get("text", "")
    metadata = doc.get("metadata", {}) or {}
    title = metadata.get("title", "")
    source_name = doc.get("source_name", "")
    url = metadata.get("url", "")

    focus_tokens, intent_tokens = _split_query_tokens(question)

    focus_body = _token_match_score(focus_tokens, text)
    focus_title = _token_match_score(focus_tokens, f"{title} {source_name} {url}")
    intent_body = _token_match_score(intent_tokens, text)
    intent_title = _token_match_score(intent_tokens, f"{title} {source_name}")
    phrase_focus = _phrase_focus_score(focus_tokens, title, source_name, url, text)

    # Premia za dokumenty, które jednocześnie pasują do jednostki chorobowej i intencji pytania.
    focus_intent_synergy = 0.0
    if focus_title >= 0.99 and (intent_title > 0 or intent_body > 0):
        focus_intent_synergy = 1.0
    elif phrase_focus >= 0.99 and (intent_title > 0 or intent_body > 0):
        focus_intent_synergy = 0.7
    elif focus_body > 0 and (intent_title > 0 or intent_body > 0):
        focus_intent_synergy = 0.3

    return {
        "focus_body": round(focus_body, 6),
        "focus_title": round(focus_title, 6),
        "intent_body": round(intent_body, 6),
        "intent_title": round(intent_title, 6),
        "phrase_focus": round(phrase_focus, 6),
        "focus_intent_synergy": round(focus_intent_synergy, 6),
    }


def retrieve_chunks(question: str, top_k: int | None = None) -> list[dict]:
    k = top_k or settings.rag_top_k
    # Pobierz więcej kandydatów niż k, żeby re-ranking miał z czego wybierać
    pre_k = max(k * 4, 20)

    query_vector = embed_texts([question])[0]

    # Brute-force KNN – działa z lokalnym MongoDB (bez Atlas)
    docs = list(collection.find(
        {"embedding": {"$exists": True}},
        {"_id": 0, "source_name": 1, "chunk_id": 1, "text": 1, "metadata": 1, "embedding": 1},
    ))

    if not docs:
        return []

    # --- batch GPU cosine similarity ---
    embeddings = [doc.pop("embedding") for doc in docs]

    # Tensory na GPU (lub CPU jeśli brak karty)
    q_t = torch.tensor(query_vector, dtype=torch.float32, device=_DEVICE).unsqueeze(0)  # (1, dim)
    e_t = torch.tensor(embeddings, dtype=torch.float32, device=_DEVICE)                 # (N, dim)

    # Wszystkie podobieństwa jedną operacją macierzową
    vec_scores: list[float] = F.cosine_similarity(q_t, e_t, dim=1).tolist()             # (N,)

    logger.debug("retrieve: batch similarity na %s, N=%d", _DEVICE, len(docs))

    scored = []
    for doc, vec_score in zip(docs, vec_scores):
        components = _rerank_components(question, doc)
        keyword_score = 0.55 * components["focus_body"] + 0.45 * components["intent_body"]
        title_score = 0.75 * components["focus_title"] + 0.25 * components["intent_title"]
        hybrid = (
            0.45 * vec_score
            + 0.20 * keyword_score
            + 0.20 * title_score
            + 0.10 * components["phrase_focus"]
            + 0.05 * components["focus_intent_synergy"]
        )
        doc["score"] = round(hybrid, 6)
        doc["_vec_score"] = round(vec_score, 6)
        doc["_kw_score"] = round(keyword_score, 6)
        doc["_title_score"] = round(title_score, 6)
        doc["_focus_body"] = components["focus_body"]
        doc["_focus_title"] = components["focus_title"]
        doc["_intent_body"] = components["intent_body"]
        doc["_intent_title"] = components["intent_title"]
        doc["_phrase_focus"] = components["phrase_focus"]
        doc["_synergy"] = components["focus_intent_synergy"]
        scored.append(doc)

    scored.sort(key=lambda d: d["score"], reverse=True)
    top = scored[:k]

    logger.debug(
        "retrieve_chunks: pytanie='%s' | kandydaci=%d | top_%d scores: %s",
        question,
        len(docs),
        k,
        [
            (
                d["chunk_id"],
                d["score"],
                d["_vec_score"],
                d["_kw_score"],
                d["_title_score"],
                d["_synergy"],
            )
            for d in top
        ],
    )
    return top
