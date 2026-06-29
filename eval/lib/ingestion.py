"""
Helpers for ingesting arXiv papers into the KnowWhere database.

Fetches metadata from the arXiv API, generates embeddings via the
KnowWhere embedding service, and inserts papers + chunks into PostgreSQL.
"""

import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import psycopg2
import requests


def _load_env() -> dict[str, str]:
    """Load key=value pairs from the project .env file."""
    env = {}
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env[key.strip()] = value.strip()
    return env


_ENV = _load_env()

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT_SEC = 12.0  # increased to avoid 429s from flagged IP
USER_AGENT = "KnowWhereEval/1.0 (https://github.com/anomalyco/knowwhere)"

_EMBEDDING_URL = _ENV.get("EMBEDDING_ENDPOINT", "http://localhost:8081/embed")
_DB_HOST = _ENV.get("DB_HOST", "localhost")
_DB_PORT = _ENV.get("DB_PORT", "5432")
_DB_NAME = _ENV.get("DB_NAME", "knowwhere")
_DB_USER = _ENV.get("DB_SUPERUSER", "knowwhere_superadmin")
_DB_PASS = _ENV.get("DB_SUPERPASS", "knowwhere_superadmin_pass")

EMBEDDING_URL = _EMBEDDING_URL
DB_DSN = (
    f"host={_DB_HOST} port={_DB_PORT} dbname={_DB_NAME} "
    f"user={_DB_USER} password={_DB_PASS}"
)
CHUNK_WORDS = 200
CHUNK_OVERLAP = 40
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _strip_version(arxiv_id: str) -> str:
    """Strip arXiv version suffix (e.g. 1703.08014v2 -> 1703.08014)."""
    m = re.match(r"^(.*?)(v\d+)$", arxiv_id)
    return m.group(1) if m else arxiv_id


def fetch_arxiv_batch(arxiv_ids: list[str]) -> list[dict]:
    """
    Fetch paper metadata from the arXiv API for a batch of IDs.

    arXiv API supports up to ~100 IDs per request via 'id_list'.
    Uses HTTPS directly (no redirect), sends User-Agent, and retries
    rate limits with progressively longer backoff.
    """
    id_str = ",".join(arxiv_ids)
    params = {"id_list": id_str, "max_results": str(len(arxiv_ids))}
    headers = {"User-Agent": USER_AGENT}

    max_retries = 5
    last_resp = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                ARXIV_API, params=params, headers=headers, timeout=90
            )
            last_resp = resp
            if resp.status_code == 429:
                wait = 10 * (2 ** attempt)  # 10, 20, 40, 80, 160s
                print(f"  Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 503:
                wait = 5 * (2 ** attempt)
                print(f"  Service unavailable (503), waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.exceptions.Timeout:
            wait = 10 * (2 ** attempt)
            print(f"  Timeout, retrying in {wait}s...")
            time.sleep(wait)
            continue
        except requests.exceptions.ConnectionError:
            wait = 10 * (2 ** attempt)
            print(f"  Connection error, retrying in {wait}s...")
            time.sleep(wait)
            continue

    if last_resp is None or last_resp.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch arXiv batch after {max_retries} attempts "
            f"(last status: {last_resp.status_code if last_resp else 'N/A'})"
        )

    root = ET.fromstring(last_resp.text)
    entries = []

    for entry in root.findall("atom:entry", NS):
        id_elem = entry.find("atom:id", NS)
        if id_elem is None:
            continue
        arxiv_id_full = id_elem.text.strip()
        arxiv_id = arxiv_id_full.split("/abs/")[-1] if "/abs/" in arxiv_id_full else arxiv_id_full

        title_elem = entry.find("atom:title", NS)
        title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""

        summary_elem = entry.find("atom:summary", NS)
        summary = (
            summary_elem.text.strip().replace("\n", " ")
            if summary_elem is not None
            else ""
        )

        pub_elem = entry.find("atom:published", NS)
        year = int(pub_elem.text.strip()[:4]) if pub_elem is not None and pub_elem.text else None

        doi_link = None
        for link in entry.findall("atom:link", NS):
            if link.attrib.get("title") == "doi":
                doi_link = link.attrib.get("href", "").strip()
                break

        authors = [
            a_elem.find("atom:name", NS).text.strip()
            for a_elem in entry.findall("atom:author", NS)
            if a_elem.find("atom:name", NS) is not None
        ]

        categories = [
            cat.attrib.get("term", "")
            for cat in entry.findall("atom:category", NS)
        ]

        entries.append(
            {
                "id": arxiv_id,
                "title": title,
                "summary": summary,
                "year": year,
                "doi": doi_link or None,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "authors": authors,
                "categories": categories,
            }
        )

    return entries


def chunk_text(
    text: str, max_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping word chunks (mirrors TypeScript chunkText)."""
    words = text.split()
    if len(words) <= max_words:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + max_words])
        if chunk.strip():
            chunks.append(chunk)
        start += max_words - overlap

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a list of texts via the KnowWhere embedding service."""
    if not texts:
        return []
    resp = requests.post(EMBEDDING_URL, json={"inputs": texts}, timeout=60)
    resp.raise_for_status()
    return resp.json()["embeddings"]


def connect_db():
    """Connect to the KnowWhere PostgreSQL database."""
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = False
    return conn


def get_existing_ids(conn, ids: list[str]) -> set[str]:
    """Return the set of paper IDs that already exist in the database."""
    cur = conn.cursor()
    cur.execute("SELECT id FROM papers WHERE id = ANY(%s)", (ids,))
    existing = {row[0] for row in cur.fetchall()}
    cur.close()
    return existing


def ensure_arxiv_source(conn) -> int:
    """Ensure the 'arxiv' source exists and return its ID."""
    cur = conn.cursor()
    cur.execute("SELECT id FROM sources WHERE name = 'arxiv'")
    row = cur.fetchone()
    if row:
        cur.close()
        return row[0]
    cur.execute(
        "INSERT INTO sources (name, base_url) "
        "VALUES ('arxiv', 'https://export.arxiv.org') "
        "RETURNING id"
    )
    source_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return source_id


def insert_paper(conn, paper: dict, embedding: list[float], source_id: int):
    """Insert a single paper with its embedding into the papers table."""
    cur = conn.cursor()
    emb_str = f"[{','.join(str(v) for v in embedding)}]"
    cur.execute(
        """
        INSERT INTO papers (id, title, abstract, authors, venue_id, year, doi, url,
                            subjects, source_id, embedding, tsv)
        VALUES (%s, %s, %s, %s, NULL, %s, %s, %s, %s, %s, %s::vector,
                to_tsvector('english', coalesce(%s,'') || ' ' || coalesce(%s,'')))
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            abstract = EXCLUDED.abstract,
            authors = EXCLUDED.authors,
            year = EXCLUDED.year,
            doi = EXCLUDED.doi,
            url = EXCLUDED.url,
            subjects = EXCLUDED.subjects,
            source_id = EXCLUDED.source_id,
            embedding = EXCLUDED.embedding,
            tsv = EXCLUDED.tsv
        """,
        (
            paper["id"],
            paper["title"],
            paper["summary"],
            paper["authors"],
            paper["year"],
            paper["doi"],
            paper["url"],
            paper["categories"],
            source_id,
            emb_str,
            paper["title"],
            paper["summary"],
        ),
    )
    conn.commit()
    cur.close()


def insert_chunks(conn, paper_id: str, summary: str):
    """Chunk a paper's summary and insert chunks with embeddings."""
    chunks = chunk_text(summary or "")
    if not chunks:
        return

    embeddings = embed_texts(chunks)
    cur = conn.cursor()
    for chunk, emb in zip(chunks, embeddings):
        emb_str = f"[{','.join(str(v) for v in emb)}]"
        cur.execute(
            """
            INSERT INTO paper_chunks (paper_id, chunk_text, chunk_embedding, tsv)
            VALUES (%s, %s, %s::vector, to_tsvector('english', coalesce(%s,'')))
            ON CONFLICT DO NOTHING
            """,
            (paper_id, chunk, emb_str, chunk),
        )
    conn.commit()
    cur.close()


def ingest_paper_batch(arxiv_ids: list[str], conn) -> dict:
    """
    Fetch, embed, and insert a batch of arXiv papers.

    Returns stats dict with counts for fetched, skipped, inserted.
    """
    stats = {"fetched": 0, "skipped": 0, "inserted": 0, "errors": 0}

    source_id = ensure_arxiv_source(conn)
    existing = get_existing_ids(conn, arxiv_ids)

    to_fetch = [aid for aid in arxiv_ids if aid not in existing]
    stats["skipped"] = len(existing)

    if not to_fetch:
        return stats

    try:
        papers = fetch_arxiv_batch(to_fetch)
        stats["fetched"] = len(papers)
    except Exception as exc:
        print(f"  arXiv fetch error: {exc}")
        stats["errors"] += 1
        return stats

    for paper in papers:
        text_for_embed = f"{paper['title']}\n{paper['summary']}"
        if not paper["summary"]:
            continue
        try:
            emb = embed_texts([text_for_embed])[0]
            insert_paper(conn, paper, emb, source_id)
            insert_chunks(conn, paper["id"], paper["summary"])
            stats["inserted"] += 1
        except Exception as exc:
            print(f"  Error inserting {paper['id']}: {exc}")
            conn.rollback()
            stats["errors"] += 1

    return stats


def ingest_all(arxiv_ids: list[str], batch_size: int = 25):
    """
    Ingest all arXiv papers in batches.

    Prints progress after each batch and a final summary.
    Returns total stats dict.
    """
    conn = connect_db()
    total = {"fetched": 0, "skipped": 0, "inserted": 0, "errors": 0}
    try:
        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(arxiv_ids) - 1) // batch_size + 1
            print(f"Batch {batch_num}/{total_batches}: {len(batch)} IDs")
            stats = ingest_paper_batch(batch, conn)
            total["fetched"] += stats["fetched"]
            total["skipped"] += stats["skipped"]
            total["inserted"] += stats["inserted"]
            total["errors"] += stats["errors"]
            print(
                f"  fetched={stats['fetched']}, skipped={stats['skipped']}, "
                f"inserted={stats['inserted']}, errors={stats['errors']}"
            )
            if i + batch_size < len(arxiv_ids):
                time.sleep(ARXIV_RATE_LIMIT_SEC)
        print(
            f"\nTotal: fetched={total['fetched']}, skipped={total['skipped']}, "
            f"inserted={total['inserted']}, errors={total['errors']}"
        )
    finally:
        conn.close()
    return total


def ingest_from_titles(id_to_title: dict[str, str], batch_size: int = 50):
    """
    Create minimal paper entries directly from arXiv ID + title pairs.

    Does NOT call the arXiv API. Uses the local embedding service to
    generate embeddings from the title text. No abstract or chunks.

    This is useful when the arXiv API is rate-limited but ground-truth
    paper IDs must exist in the database for evaluation matching.
    """
    conn = connect_db()
    source_id = ensure_arxiv_source(conn)
    all_ids = sorted(id_to_title.keys())
    existing = get_existing_ids(conn, all_ids)
    to_insert = [aid for aid in all_ids if aid not in existing]

    if not to_insert:
        print("All papers already in DB.")
        conn.close()
        return {"fetched": 0, "skipped": len(existing), "inserted": 0, "errors": 0}

    print(f"Creating {len(to_insert)} minimal paper entries "
          f"(skipping {len(existing)} already in DB)...")

    total = {"fetched": 0, "skipped": len(existing), "inserted": 0, "errors": 0}

    try:
        for i in range(0, len(to_insert), batch_size):
            batch = to_insert[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(to_insert) - 1) // batch_size + 1

            # Collect titles for this batch
            titles = [id_to_title[aid] for aid in batch]
            texts = [f"{id_to_title[aid]}" for aid in batch]

            # Embed all titles in one call
            embeddings = embed_texts(texts)

            cur = conn.cursor()
            inserted = 0
            for aid, title, emb in zip(batch, titles, embeddings):
                try:
                    emb_str = f"[{','.join(str(v) for v in emb)}]"
                    cur.execute(
                        """
                        INSERT INTO papers (id, title, abstract, authors, venue_id,
                                            year, doi, url, subjects, source_id,
                                            embedding, tsv)
                        VALUES (%s, %s, %s, %s, NULL, NULL, NULL, %s, %s, %s,
                                %s::vector,
                                to_tsvector('english', coalesce(%s,'') || ' ' || coalesce(%s,'')))
                        ON CONFLICT (id) DO UPDATE SET
                            title = EXCLUDED.title,
                            abstract = EXCLUDED.abstract,
                            url = EXCLUDED.url,
                            source_id = EXCLUDED.source_id,
                            embedding = EXCLUDED.embedding,
                            tsv = EXCLUDED.tsv
                        """,
                        (
                            aid,
                            title,
                            title,  # abstract = title (minimal)
                            [],
                            f"https://arxiv.org/abs/{aid}",
                            [],
                            source_id,
                            emb_str,
                            title,
                            title,
                        ),
                    )
                    inserted += 1
                except Exception as exc:
                    print(f"  Error inserting {aid}: {exc}")
                    conn.rollback()
                    total["errors"] += 1

            conn.commit()
            cur.close()
            total["fetched"] += len(batch)
            total["inserted"] += inserted
            print(f"  Batch {batch_num}/{total_batches}: inserted {inserted}/{len(batch)}")

    finally:
        conn.close()

    print(f"Total: inserted={total['inserted']}, skipped={total['skipped']}, errors={total['errors']}")
    return total


def enrich_abstracts(arxiv_ids: list[str], batch_size: int = 10):
    """
    Fetch real abstracts from the arXiv API and update existing paper rows.

    Updates: abstract, tsv, embedding (title + abstract), and inserts chunks.
    Uses HTTPS with User-Agent and rate-limit-aware delays.
    """
    conn = connect_db()
    total_batches = (len(arxiv_ids) - 1) // batch_size + 1
    total_papers = len(arxiv_ids)
    total = {"fetched": 0, "updated": 0, "enriched": 0, "skipped": 0, "errors": 0}
    start_time = time.time()

    def _progress():
        elapsed = time.time() - start_time
        done = total["enriched"]
        pct = done / max(total_papers, 1) * 100
        rate = done / max(elapsed, 1)
        eta = (total_papers - done) / max(rate, 0.001)
        print(
            f"  [{done}/{total_papers} {pct:.1f}%] "
            f"rate={rate:.1f}/s, eta={eta:.0f}s"
        )

    try:
        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i : i + batch_size]
            batch_num = i // batch_size + 1

            try:
                papers = fetch_arxiv_batch(batch)
                total["fetched"] += len(papers)
            except Exception as exc:
                print(f"  Batch {batch_num}/{total_batches}: arXiv error: {exc}")
                total["errors"] += 1
                _progress()
                if i + batch_size < len(arxiv_ids):
                    time.sleep(ARXIV_RATE_LIMIT_SEC)
                continue

            for paper in papers:
                total["enriched"] += 1
                if not paper["summary"] or len(paper["summary"]) < 50:
                    total["skipped"] += 1
                    continue

                clean_id = _strip_version(paper["id"])
                text = f"{paper['title']}\n{paper['summary']}"
                try:
                    emb = embed_texts([text])[0]
                    emb_str = f"[{','.join(str(v) for v in emb)}]"

                    cur = conn.cursor()
                    cur.execute(
                        """
                        UPDATE papers
                        SET abstract = %s,
                            embedding = %s::vector,
                            tsv = to_tsvector('english', coalesce(%s,'') || ' ' || coalesce(%s,''))
                        WHERE id = %s
                        """,
                        (paper["summary"], emb_str, paper["title"], paper["summary"], clean_id),
                    )
                    conn.commit()
                    cur.close()

                    try:
                        insert_chunks(conn, clean_id, paper["summary"])
                    except Exception:
                        conn.rollback()
                    total["updated"] += 1
                except Exception as exc:
                    print(f"\n  Error enriching {clean_id}: {exc}")
                    conn.rollback()
                    total["errors"] += 1

            print(
                f"  Batch {batch_num}/{total_batches}: "
                f"fetched={len(papers)} ({total['updated']} updated)"
            )
            _progress()

            if i + batch_size < len(arxiv_ids):
                time.sleep(ARXIV_RATE_LIMIT_SEC)

    finally:
        conn.close()

    elapsed = time.time() - start_time
    print(
        f"\nDone in {elapsed:.0f}s: fetched={total['fetched']}, "
        f"updated={total['updated']}, enriched={total['enriched']}, "
        f"skipped={total['skipped']}, errors={total['errors']}"
    )
    return total


def enrich_via_openalex(arxiv_ids: list[str]):
    """
    Fetch abstracts from OpenAlex (no rate limits on free tier: 10 req/s).

    Uses the DOI prefix 10.48550/arxiv.{id} to query OpenAlex.
    Strips arXiv version suffixes before querying.
    Updates: abstract, tsv, embedding, and chunks.
    """
    conn = connect_db()
    total = {"found": 0, "updated": 0, "skipped": 0, "errors": 0}
    start_time = time.time()

    try:
        for i, aid in enumerate(arxiv_ids):
            clean_id = _strip_version(aid)
            url = f"https://api.openalex.org/works?filter=doi:10.48550/arxiv.{clean_id}&select=title,abstract_inverted_index"
            try:
                resp = requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
                data = resp.json()
            except Exception as exc:
                total["errors"] += 1
                continue

            results = data.get("results", [])
            if not results:
                total["skipped"] += 1
                continue

            inv = results[0].get("abstract_inverted_index")
            if not inv:
                total["skipped"] += 1
                continue

            words = sorted(inv.items(), key=lambda x: x[1][0])
            abstract = " ".join(w for w, _ in words)

            if len(abstract) < 50:
                total["skipped"] += 1
                continue

            total["found"] += 1

            title = results[0].get("title", "Unknown")
            text = f"{title}\n{abstract}"
            try:
                emb = embed_texts([text])[0]
                emb_str = f"[{','.join(str(v) for v in emb)}]"

                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE papers
                    SET abstract = %s,
                        title = %s,
                        embedding = %s::vector,
                        tsv = to_tsvector('english', coalesce(%s,'') || ' ' || coalesce(%s,''))
                    WHERE id = %s
                    """,
                    (abstract, title, emb_str, title, abstract, clean_id),
                )
                conn.commit()
                cur.close()

                try:
                    insert_chunks(conn, clean_id, abstract)
                except Exception:
                    conn.rollback()

                total["updated"] += 1
            except Exception as exc:
                conn.rollback()
                total["errors"] += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(arxiv_ids):
                elapsed = time.time() - start_time
                pct = (i + 1) / len(arxiv_ids) * 100
                print(
                    f"  [{i+1}/{len(arxiv_ids)} {pct:.0f}%] "
                    f"found={total['found']} updated={total['updated']} "
                    f"skipped={total['skipped']} errors={total['errors']} "
                    f"elapsed={elapsed:.0f}s"
                )

            time.sleep(0.2)

    finally:
        conn.close()

    elapsed = time.time() - start_time
    print(
        f"\nDone in {elapsed:.0f}s: found={total['found']} updated={total['updated']} "
        f"skipped={total['skipped']} errors={total['errors']}"
    )
    return total
