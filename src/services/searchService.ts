import { query } from '../db/db';
import { embedText } from './embeddingClient';
import { rerank } from './rerankService';
import { Level, Mode, SearchFilters, SearchRequest, SearchResult } from '../types';
import {
  SEARCH_LIMITS,
  SEARCH_CANDIDATES,
  SEARCH_QUALITY,
  SEARCH_WEIGHTS,
  SNIPPET_LENGTH
} from '../config/search/constants';

function vectorLiteral(vec: number[]): string {
  if (!Array.isArray(vec) || vec.length === 0) {
    throw new Error('Embedding vector missing');
  }
  return `[${vec.join(',')}]`;
}

function buildFilters(
  alias: string,
  filters: SearchFilters | undefined,
  values: unknown[]
): string {
  const clauses: string[] = [];
  if (!filters) return '';

  if (filters.yearFrom) {
    values.push(filters.yearFrom);
    clauses.push(`${alias}.year >= $${values.length}`);
  }
  if (filters.yearTo) {
    values.push(filters.yearTo);
    clauses.push(`${alias}.year <= $${values.length}`);
  }
  if (filters.venue) {
    values.push(filters.venue);
    clauses.push(`v.name ILIKE $${values.length}`);
  }
  if (filters.subject) {
    values.push(filters.subject);
    clauses.push(`${alias}.subjects @> ARRAY[$${values.length}]::text[]`);
  }
  if (filters.source) {
    values.push(filters.source);
    clauses.push(`s.name = $${values.length}`);
  }

  return clauses.length ? ` AND ${clauses.join(' AND ')}` : '';
}

interface PaperRow {
  id: string;
  title: string | null;
  abstract: string | null;
  doi: string | null;
  url: string | null;
  subjects: string[] | null;
  source: string | null;
  snippet: string | null;
  lex_score?: number | null;
  sem_score?: number | null;
  hybrid_score: number | null;
}

interface ChunkRow extends PaperRow {
  chunk_id: number | null;
}

function retrievalStrength(r: SearchResult): number {
  const v = r.hybridScore ?? r.lexScore ?? r.semScore;
  return v != null && Number.isFinite(Number(v)) ? Number(v) : Number.NEGATIVE_INFINITY;
}

function normalizeDedupeTitle(title: string | null | undefined): string {
  if (!title) return '';
  return title.trim().toLowerCase().replace(/\s+/g, ' ');
}

/**
 * Drop duplicate catalog rows: same DOI, then same normalized title (keep best retrieval score).
 * Preserves first-seen order of surviving keys.
 */
function dedupePaperSearchResults(results: SearchResult[]): SearchResult[] {
  const mergeByKey = (rows: SearchResult[], keyFn: (row: SearchResult) => string): SearchResult[] => {
    const keyOrder: string[] = [];
    const best = new Map<string, SearchResult>();
    for (const row of rows) {
      const k = keyFn(row);
      const prev = best.get(k);
      if (!prev) {
        keyOrder.push(k);
        best.set(k, row);
      } else if (retrievalStrength(row) > retrievalStrength(prev)) {
        best.set(k, row);
      }
    }
    return keyOrder.map((k) => best.get(k)!);
  };

  const doiKey = (row: SearchResult) => {
    const d = row.doi?.trim().toLowerCase();
    return d ? `doi:${d}` : `\0id:${row.id}`;
  };
  const titleKey = (row: SearchResult) => {
    const t = normalizeDedupeTitle(row.title);
    return t ? `title:${t}` : `\0id:${row.id}`;
  };

  return mergeByKey(mergeByKey(results, doiKey), titleKey);
}

function hasLexicalHint(r: SearchResult): boolean {
  return r.lexScore != null && Number(r.lexScore) > 1e-6;
}

function isStubMetadata(r: SearchResult): boolean {
  const title = (r.title ?? '').trim();
  const abs = (r.abstract ?? '').trim();
  if (!abs) return true;
  if (abs.length < SEARCH_QUALITY.stubAbstractMaxLen) return true;
  if (title && abs.toLowerCase() === title.toLowerCase()) return true;
  return false;
}

/** Tweak fused hybrid scores before dedupe/rerank so substantive papers beat semantic-only stubs. */
function hybridMetadataMultiplier(r: SearchResult): number {
  const hasLex = hasLexicalHint(r);
  const stub = isStubMetadata(r);
  // Stub catalog rows stay penalized even when they lexically match generic query terms ("machine learning").
  if (stub) {
    return SEARCH_QUALITY.stubPenalty * SEARCH_QUALITY.semanticOnlyStubExtra;
  }
  if (hasLex) return SEARCH_QUALITY.lexicalSemanticBoost;
  return 1;
}

function applyPaperRetrievalHeuristic(results: SearchResult[], mode: Mode): SearchResult[] {
  if (mode !== 'hybrid' && mode !== 'semantic') return results;
  const tweaked = results.map((row) => {
    const raw = row.hybridScore;
    if (raw == null || !Number.isFinite(Number(raw))) return row;
    const mult = hybridMetadataMultiplier(row);
    const next = Math.min(1, Math.max(0, Number(raw) * mult));
    return { ...row, hybridScore: next };
  });
  return tweaked.sort((a, b) => (b.hybridScore ?? 0) - (a.hybridScore ?? 0));
}

export async function search(request: SearchRequest): Promise<SearchResult[]> {
  const limit = Math.min(request.limit ?? SEARCH_LIMITS.default, SEARCH_LIMITS.max);
  const mode: Mode = request.mode ?? 'hybrid';
  const level: Level = request.level ?? 'paper';

  const needsEmbedding = mode !== 'lexical';
  const embedding = needsEmbedding ? await embedText(request.q) : null;

  const { sql, values } =
    level === 'chunk'
      ? buildChunkQuery(request.q, mode, request.filters, limit, embedding)
      : buildPaperQuery(request.q, mode, request.filters, limit, embedding);

  if (process.env.DEBUG_SEARCH) {
    // eslint-disable-next-line no-console
    console.log('[search debug]', { mode, level, filters: request.filters, sql, values });
  }

  const { rows } = await query<PaperRow | ChunkRow>(sql, values);

  let baseResults: SearchResult[] = rows.map((row: PaperRow | ChunkRow) => ({
    id: row.id,
    title: row.title,
    abstract: row.abstract,
    doi: row.doi,
    url: row.url,
    subjects: row.subjects,
    source: row.source,
    snippet: row.snippet,
    lexScore: 'lex_score' in row ? row.lex_score ?? undefined : undefined,
    semScore: 'sem_score' in row ? row.sem_score ?? undefined : undefined,
    hybridScore: row.hybrid_score,
    chunkId: 'chunk_id' in row ? row.chunk_id ?? undefined : undefined
  }));

  if (level === 'paper') {
    baseResults = applyPaperRetrievalHeuristic(baseResults, mode);
    baseResults = dedupePaperSearchResults(baseResults);
  }

  // Always rerank in normal operation; tests may bypass via SKIP_RERANK.
  const rerankTop = Math.min(baseResults.length, 20);
  const substantives = baseResults.filter((r) => !isStubMetadata(r));
  const rerankCandidates =
    substantives.length > 0 ? substantives.slice(0, rerankTop) : baseResults.slice(0, rerankTop);
  const rerankCandidateIds = new Set(rerankCandidates.map((r) => r.id));
  let rerankScores: number[] | null = null;
  if (rerankCandidates.length > 0) {
    try {
      rerankScores = await rerank(
        request.q,
        rerankCandidates.map((r) => r.snippet ?? r.abstract ?? r.title ?? '')
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      // eslint-disable-next-line no-console
      console.error('[search] rerank failed; returning fusion-ranked results', { message });
      rerankScores = null;
    }
  }

  if (rerankScores) {
    const scored = rerankCandidates.map((result, idx) => ({
      result,
      score: rerankScores[idx] ?? 0
    }));

    // Final relevance gate: only return results that are actually about the query.
    // This intentionally allows returning an empty list when confidence is low.
    const gateEnabled = !(process.env.RERANK_ABSTAIN === '0' || process.env.RERANK_ABSTAIN === 'false');
    const minTop =
      process.env.RERANK_ABSTAIN_MIN_TOP !== undefined ? Number(process.env.RERANK_ABSTAIN_MIN_TOP) : 0.2;
    const maxDrop =
      process.env.RERANK_ABSTAIN_MAX_DROP !== undefined ? Number(process.env.RERANK_ABSTAIN_MAX_DROP) : 0.25;
    const minGap =
      process.env.RERANK_ABSTAIN_MIN_GAP !== undefined ? Number(process.env.RERANK_ABSTAIN_MIN_GAP) : 0.05;

    const scoredSorted = scored.sort((a, b) => b.score - a.score);
    const topScore = scoredSorted[0]?.score ?? -Infinity;
    const secondScore = scoredSorted[1]?.score ?? -Infinity;
    const gap = topScore - secondScore;

    const reranked =
      gateEnabled && Number.isFinite(topScore) && Number.isFinite(minTop) && topScore < minTop
        ? []
        : gateEnabled && Number.isFinite(gap) && Number.isFinite(minGap) && scoredSorted.length > 1 && gap < minGap
          ? []
          : gateEnabled && Number.isFinite(maxDrop)
            ? scoredSorted.filter((s) => s.score >= topScore - maxDrop).map(({ result }) => result)
            : scoredSorted.map(({ result }) => result);

    if (gateEnabled && reranked.length === 0) {
      // Long paraphrased queries often yield uniformly low rerank scores vs 240-char snippets;
      // still return fusion-ranked candidates rather than an empty list (set RERANK_ABSTAIN=0 to skip gating).
      const remainder = baseResults.filter((r) => !rerankCandidateIds.has(r.id));
      return [...rerankCandidates, ...remainder];
    }
    const remainder = baseResults.filter((r) => !rerankCandidateIds.has(r.id));
    return [...reranked, ...remainder];
  }

  return baseResults;
}

function buildPaperQuery(
  q: string,
  mode: Mode,
  filters: SearchFilters | undefined,
  limit: number,
  embedding: number[] | null
) {
  const values: unknown[] = [q];

  if (mode === 'lexical') {
    const filterSql = buildFilters('p', filters, values);
    values.push(limit);
    const sql = `
      WITH q AS (
        SELECT websearch_to_tsquery('english', $1) AS q_ts
      )
      SELECT p.id,
             p.title,
             p.abstract,
             p.doi,
             p.url,
             p.subjects,
             s.name AS source,
             LEFT(p.abstract, ${SNIPPET_LENGTH}) AS snippet,
             ts_rank_cd(p.tsv, q.q_ts) AS lex_score,
             NULL::float AS sem_score,
             ts_rank_cd(p.tsv, q.q_ts) AS hybrid_score
      FROM papers p
      LEFT JOIN sources s ON p.source_id = s.id
      LEFT JOIN venues v ON p.venue_id = v.id,
           q
      WHERE q.q_ts <> '' AND p.tsv @@ q.q_ts${filterSql}
      ORDER BY lex_score DESC
      LIMIT $${values.length};
    `;
    return { sql, values };
  }

  const vectorParam = vectorLiteral(embedding ?? []);

  if (mode === 'semantic') {
    const semanticValues: unknown[] = [];
    const vectorIdx = semanticValues.push(vectorParam);
    const filterSql = buildFilters('p', filters, semanticValues);
    semanticValues.push(limit);
    const vectorRef = `$${vectorIdx}`;
    const sql = `
      SELECT p.id,
             p.title,
             p.abstract,
             p.doi,
             p.url,
             p.subjects,
             s.name AS source,
             LEFT(p.abstract, ${SNIPPET_LENGTH}) AS snippet,
             NULL::float AS lex_score,
             1 - (p.embedding <=> ${vectorRef}::vector) AS sem_score,
             1 - (p.embedding <=> ${vectorRef}::vector) AS hybrid_score
      FROM papers p
      LEFT JOIN sources s ON p.source_id = s.id
      LEFT JOIN venues v ON p.venue_id = v.id
      WHERE p.embedding IS NOT NULL${filterSql}
      ORDER BY p.embedding <=> ${vectorRef}::vector
      LIMIT $${semanticValues.length};
    `;
    return { sql, values: semanticValues };
  }

  // hybrid
  const filterSql = buildFilters('p', filters, values);
  const vectorIdx = values.push(vectorParam);
  values.push(limit);
  const limitParam = values.length;
  const vectorRef = `$${vectorIdx}`;
  const sql = `
    WITH q AS (
      SELECT
        websearch_to_tsquery('english', $1) AS q_ts,
        ${vectorRef}::vector AS q_vec
    ),
    lex AS (
      SELECT p.id, p.title, p.abstract, p.doi, p.url, p.subjects, s.name AS source,
             LEFT(p.abstract, ${SNIPPET_LENGTH}) AS snippet,
             ts_rank_cd(p.tsv, q.q_ts) AS lex_score,
             NULL::float AS sem_score
      FROM papers p
      LEFT JOIN sources s ON p.source_id = s.id
      LEFT JOIN venues v ON p.venue_id = v.id,
           q
      WHERE q.q_ts <> '' AND p.tsv @@ q.q_ts${filterSql}
      ORDER BY lex_score DESC
      LIMIT ${SEARCH_CANDIDATES.hybridLexical}
    ),
    sem AS (
      SELECT p.id, p.title, p.abstract, p.doi, p.url, p.subjects, s.name AS source,
             LEFT(p.abstract, ${SNIPPET_LENGTH}) AS snippet,
             NULL::float AS lex_score,
             1 - (p.embedding <=> q.q_vec) AS sem_score
      FROM papers p
      LEFT JOIN sources s ON p.source_id = s.id
      LEFT JOIN venues v ON p.venue_id = v.id,
           q
      WHERE p.embedding IS NOT NULL${filterSql}
      ORDER BY p.embedding <=> q.q_vec
      LIMIT ${SEARCH_CANDIDATES.hybridSemantic}
    ),
    fused AS (
      SELECT id, title, abstract, doi, url, subjects, source, snippet,
             max(lex_score) AS lex_score,
             max(sem_score) AS sem_score,
             coalesce(max(lex_score), 0) * ${SEARCH_WEIGHTS.lexical} + coalesce(max(sem_score), 0) * ${SEARCH_WEIGHTS.semantic} AS hybrid_score
      FROM (
        SELECT * FROM lex
        UNION ALL
        SELECT * FROM sem
      ) s
      GROUP BY id, title, abstract, doi, url, subjects, source, snippet
    )
    SELECT * FROM fused
    ORDER BY hybrid_score DESC
    LIMIT $${limitParam};
  `;
  return { sql, values };
}

function buildChunkQuery(
  q: string,
  mode: Mode,
  filters: SearchFilters | undefined,
  limit: number,
  embedding: number[] | null
) {
  const values: unknown[] = [q];

  if (mode === 'lexical') {
    const filterSql = buildFilters('p', filters, values);
    values.push(limit);
    const sql = `
      WITH q AS (
        SELECT websearch_to_tsquery('english', $1) AS q_ts
      )
      SELECT p.id,
             p.title,
             p.abstract,
             p.doi,
             p.url,
             p.subjects,
             s.name AS source,
             c.chunk_id,
             LEFT(c.chunk_text, ${SNIPPET_LENGTH}) AS snippet,
             ts_rank_cd(c.tsv, q.q_ts) AS lex_score,
             NULL::float AS sem_score,
             ts_rank_cd(c.tsv, q.q_ts) AS hybrid_score
      FROM paper_chunks c
      JOIN papers p ON c.paper_id = p.id
      LEFT JOIN sources s ON p.source_id = s.id
      LEFT JOIN venues v ON p.venue_id = v.id,
           q
      WHERE q.q_ts <> '' AND c.tsv @@ q.q_ts${filterSql}
      ORDER BY lex_score DESC
      LIMIT $${values.length};
    `;
    return { sql, values };
  }

  const vectorParam = vectorLiteral(embedding ?? []);

  if (mode === 'semantic') {
    const semanticValues: unknown[] = [];
    const vectorIdx = semanticValues.push(vectorParam);
    const filterSql = buildFilters('p', filters, semanticValues);
    semanticValues.push(limit);
    const vectorRef = `$${vectorIdx}`;
    const sql = `
      SELECT p.id,
             p.title,
             p.abstract,
             p.doi,
             p.url,
             p.subjects,
             s.name AS source,
             c.chunk_id,
             LEFT(c.chunk_text, ${SNIPPET_LENGTH}) AS snippet,
             NULL::float AS lex_score,
             1 - (c.chunk_embedding <=> ${vectorRef}::vector) AS sem_score,
             1 - (c.chunk_embedding <=> ${vectorRef}::vector) AS hybrid_score
      FROM paper_chunks c
      JOIN papers p ON c.paper_id = p.id
      LEFT JOIN sources s ON p.source_id = s.id
      LEFT JOIN venues v ON p.venue_id = v.id
      WHERE c.chunk_embedding IS NOT NULL${filterSql}
      ORDER BY c.chunk_embedding <=> ${vectorRef}::vector
      LIMIT $${semanticValues.length};
    `;
    return { sql, values: semanticValues };
  }

  // hybrid
  const filterSql = buildFilters('p', filters, values);
  const vectorIdx = values.push(vectorParam);
  values.push(limit);
  const limitParam = values.length;
  const vectorRef = `$${vectorIdx}`;
  const sql = `
    WITH q AS (
      SELECT
        websearch_to_tsquery('english', $1) AS q_ts,
        ${vectorRef}::vector AS q_vec
    ),
    lex AS (
      SELECT p.id, p.title, p.abstract, p.doi, p.url, p.subjects, s.name AS source,
             c.chunk_id,
             LEFT(c.chunk_text, ${SNIPPET_LENGTH}) AS snippet,
             ts_rank_cd(c.tsv, q.q_ts) AS lex_score,
             NULL::float AS sem_score
      FROM paper_chunks c
      JOIN papers p ON c.paper_id = p.id
      LEFT JOIN sources s ON p.source_id = s.id
      LEFT JOIN venues v ON p.venue_id = v.id,
           q
      WHERE q.q_ts <> '' AND c.tsv @@ q.q_ts${filterSql}
      ORDER BY lex_score DESC
      LIMIT ${SEARCH_CANDIDATES.hybridLexical}
    ),
    sem AS (
      SELECT p.id, p.title, p.abstract, p.doi, p.url, p.subjects, s.name AS source,
             c.chunk_id,
             LEFT(c.chunk_text, ${SNIPPET_LENGTH}) AS snippet,
             NULL::float AS lex_score,
             1 - (c.chunk_embedding <=> q.q_vec) AS sem_score
      FROM paper_chunks c
      JOIN papers p ON c.paper_id = p.id
      LEFT JOIN sources s ON p.source_id = s.id
      LEFT JOIN venues v ON p.venue_id = v.id,
           q
      WHERE c.chunk_embedding IS NOT NULL${filterSql}
      ORDER BY c.chunk_embedding <=> q.q_vec
      LIMIT ${SEARCH_CANDIDATES.hybridSemantic}
    ),
    fused AS (
      SELECT id, title, abstract, doi, url, subjects, source, chunk_id, snippet,
             max(lex_score) AS lex_score,
             max(sem_score) AS sem_score,
             coalesce(max(lex_score), 0) * ${SEARCH_WEIGHTS.lexical} + coalesce(max(sem_score), 0) * ${SEARCH_WEIGHTS.semantic} AS hybrid_score
      FROM (
        SELECT * FROM lex
        UNION ALL
        SELECT * FROM sem
      ) s
      GROUP BY id, title, abstract, doi, url, subjects, source, chunk_id, snippet
    )
    SELECT * FROM fused
    ORDER BY hybrid_score DESC
    LIMIT $${limitParam};
  `;
  return { sql, values };
}
