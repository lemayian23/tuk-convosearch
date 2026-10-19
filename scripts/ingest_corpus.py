#!/usr/bin/env python
"""
Build or refresh the TU-K search index from the corpus directory.

    python scripts/ingest_corpus.py                 # index anything new/changed
    python scripts/ingest_corpus.py --force         # re-embed every file
    python scripts/ingest_corpus.py --corpus-dir /path/to/files

Every chunk's embedding is stored in SQLite; the FAISS structure is derived
from it at boot, so this script only has to write rows. --force is what you
run after changing TUK_CHUNK_SIZE or TUK_EMBEDDING_MODEL, because those
change the vectors rather than the document list.

Location: scripts/ingest_corpus.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.ingestion.documents import SUPPORTED_EXTENSIONS  # noqa: E402
from app.ingestion.index import CorpusIndex  # noqa: E402
from app.services import database  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the TU-K corpus")
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete all chunks first and re-embed every file",
    )
    args = parser.parse_args()

    settings = get_settings()
    corpus_dir = args.corpus_dir or settings.corpus_dir
    database.init_db()

    if not corpus_dir.exists():
        print(f"Corpus directory does not exist: {corpus_dir}")
        return 1

    files = sorted(p for p in corpus_dir.glob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    skipped = sorted(
        p.name
        for p in corpus_dir.glob("*")
        if p.is_file() and p.suffix.lower() not in SUPPORTED_EXTENSIONS
    )
    if not files:
        print(f"No supported files found in {corpus_dir}")
        return 1

    index = CorpusIndex(settings)
    if args.force:
        print("--force: dropping all stored vectors")
        index.wipe()

    print(f"Ingesting {len(files)} file(s) from {corpus_dir}")
    failures = 0
    for path in files:
        try:
            result = index.ingest_path(path)
        except Exception as exc:
            print(f"  ✗ {path.name}: {exc}")
            failures += 1
            continue
        print(
            f"  {result['status']:>9}  {path.name}"
            f"  ({result.get('chunks', 0)} chunks)"
        )

    if skipped:
        print(f"Skipped unsupported: {', '.join(skipped)}")

    stats = index.get_stats()
    print(f"\nIndex: {stats['total_chunks']} chunks, {stats['embedding_model']}")
    print(
        "NOTE: this ran in its own process, so the API is still answering from the\n"
        "index it loaded at boot. Restart it to pick up these vectors:\n"
        "    docker compose --project-directory . -f deploy/docker-compose.yml restart app\n"
        "Uploads through the admin panel do not need this; they reload in-process."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
