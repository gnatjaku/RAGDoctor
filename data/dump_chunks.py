#!/usr/bin/env python3
"""
Pobiera wszystkie chunki z MongoDB, skleja je w tekst i zapisuje do pliku.
Użycie:
    python dump_chunks.py [--source source_id] [--out output.txt]
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(Path(__file__).parent.parent / ".env")

MONGO_URI = os.getenv("MONGODB_URI") or (
    f"mongodb://{os.getenv('MONGO_INITDB_ROOT_USERNAME')}:"
    f"{os.getenv('MONGO_INITDB_ROOT_PASSWORD')}@localhost:27017"
)
DB_NAME  = os.getenv("MONGODB_DB", "rag_db")
COL_NAME = os.getenv("MONGODB_COLLECTION", "documents")

SEP = "\n" + "─" * 80 + "\n"


def dump(source_id: str | None, out_path: str | None) -> None:
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COL_NAME]

    query = {"source_id": source_id} if source_id else {}
    total = col.count_documents(query)

    if total == 0:
        print("[dump] Brak dokumentów w bazie.", file=sys.stderr)
        sys.exit(1)

    print(f"[dump] Znaleziono {total} chunków"
          + (f" dla source_id={source_id}" if source_id else "") + ".")

    # Grupuj po source_id, sortuj po chunk_id
    docs = list(
        col.find(query, {"_id": 0, "source_id": 1, "source_name": 1, "chunk_id": 1, "text": 1})
           .sort("chunk_id", 1)
    )

    # Grupuj po source
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for doc in docs:
        groups[doc.get("source_id", "unknown")].append(doc)

    lines = []
    for src_id, chunks in sorted(groups.items()):
        src_name = chunks[0].get("source_name", src_id)
        lines.append(f"{'═' * 80}")
        lines.append(f"SOURCE ID  : {src_id}")
        lines.append(f"SOURCE NAME: {src_name}")
        lines.append(f"CHUNKI     : {len(chunks)}")
        lines.append(f"{'═' * 80}")
        for chunk in chunks:
            lines.append(f"\n[chunk_id={chunk.get('chunk_id')}]")
            lines.append(chunk.get("text", "").strip())
            lines.append("─" * 80)

    full_text = "\n".join(lines)

    if out_path:
        Path(out_path).write_text(full_text, encoding="utf-8")
        print(f"[dump] Zapisano do: {out_path}  ({len(full_text)} znaków)")
    else:
        # Wypisz na stdout
        sys.stdout.buffer.write(full_text.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
        print(f"\n[dump] Łącznie znaków: {len(full_text)}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sklej chunki z MongoDB w tekst")
    parser.add_argument("--source", default=None, help="Filtruj po source_id (domyślnie: wszystkie)")
    parser.add_argument("--out",    default=None, help="Plik wyjściowy (domyślnie: stdout)")
    args = parser.parse_args()

    dump(source_id=args.source, out_path=args.out)


if __name__ == "__main__":
    main()
