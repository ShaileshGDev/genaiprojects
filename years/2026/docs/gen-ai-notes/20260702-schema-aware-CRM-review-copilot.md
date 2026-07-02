Yes — but don’t “train” the local LLM on your CRM database objects in the classic fine-tuning sense first. For your use case, the highest-value design is a **schema-aware RAG + rules + tool-calling reviewer** that knows your real database objects, business glossary, dependencies, and query patterns, then uses that context during review time. That approach is usually better than raw fine-tuning because it keeps the assistant current, grounded in your real schema, and able to review business logic instead of only style issues. [arxiv](https://arxiv.org/abs/2511.05302)

## Recommended approach

Build a local review assistant with four layers: a **knowledge layer** for schema and logic, a retrieval layer for relevant context, a rules layer for your checklist, and an execution layer for safe metadata lookup. Retrieval-augmented review works well because external domain knowledge improves review quality and keeps comments specific instead of generic. For a PostgreSQL-centric setup, pgvector lets you store embeddings in Postgres and query them with similarity search, which is a strong fit when you already run Postgres and want relational joins plus semantic retrieval in one stack. [brandonwie](https://brandonwie.dev/posts/pgvector-hnsw-postgresql)

A practical stack for your environment is: local LLM via Ollama/vLLM, embeddings stored in PostgreSQL with pgvector, ingestion scripts in Python, and a thin MCP or tool layer to expose approved schema/lineage/lookups to the model. MCP-style tooling helps keep the model from hallucinating by making it call explicit tools for database facts instead of inventing them from prompt context alone. [enterprisedb](https://www.enterprisedb.com/blog/building-real-time-data-aware-intelligence-postgres-and-model-context-protocol)

## What to ingest

Do not embed only raw SQL files. Ingest a richer knowledge graph of your CRM database landscape: stored procedures, functions, views, tables, indexes, job definitions, migration history, column descriptions, foreign-key relationships, and business meanings like “lead,” “distributor,” “secondary sale,” “claim,” or “beat plan.” Storing embeddings separately from primary tables is a common pgvector best practice because vectors are large, rarely needed in regular OLTP queries, and easier to refresh independently; storing a content hash also helps you skip unnecessary re-embedding when object text has not changed. [brandonwie](https://brandonwie.dev/ko/posts/pgvector-hnsw-postgresql)

For each object, create multiple retrievable artifacts instead of one blob:
- Raw SQL text.
- Normalized SQL text.
- One-paragraph LLM-generated summary.
- Dependency metadata, object type, schema, referenced tables, referenced columns.
- Business-domain tags, such as CRM, orders, incentives, collections, territory mapping.
- Review history and known incidents tied to that object.

This matters because your reviewer should answer questions like: “This proc updates invoice aging — does it align with actual receivables logic?” and “This view is technically correct, but is it violating our distributor-credit business rule?” Those answers require business context, not just syntax.

## Architecture

Use a hybrid retrieval pipeline, not embeddings alone. Start with lexical lookup on object name, schema, and identifiers, then add semantic retrieval over summaries and business descriptions, then enrich with dependency traversal across related tables/views/procs. Pure semantic search is often weak for exact identifiers like `usp_upsert_lead_score`, while pure keyword search misses conceptual matches like “credit exposure” versus “outstanding amount”. [postgresql.fastware](https://www.postgresql.fastware.com/blog/how-to-store-and-query-embeddings-in-postgresql-without-losing-your-mind)

A good pipeline at review time looks like this:
1. Input SQL object or PR diff.
2. Detect object type, touched tables, columns, joins, and keywords.
3. Retrieve:
   - same object’s older versions,
   - dependent/upstream/downstream objects,
   - business glossary entries,
   - relevant checklist rules,
   - similar prior review comments or incidents.
4. Give the model structured context.
5. Ask it to review against both technical and domain rules.
6. Return findings with severity, rationale, and suggested fixes.

This gives you a reviewer that can say, for example: “This join is performant, but the logic likely double-counts secondary sales because returns are stored separately and must be netted before monthly aggregation,” which is where actual value appears.

## “Train” vs RAG

Use fine-tuning only later, and only for output style or internal review tone. Fine-tuning is weaker for a fast-changing CRM schema because your objects, procedures, and business rules change often, and retraining every time is expensive and stale-prone. RAG is better for current knowledge; fine-tuning is better for stable behavior, such as always producing severity-ranked findings in your preferred format. [arxiv](https://arxiv.org/abs/2511.05302)

A good split is:
- **RAG** for current schema, object logic, incidents, naming standards, checklists.
- **Fine-tuning or instruction tuning** for response format, tone, and consistent categorization.
- **Tool calls** for live metadata, dependency lookup, sample plans, and object diffs.

## Data model to build

Use Postgres as your control plane. Create tables like:
- `db_objects` — object id, type, schema, name, source_sql, normalized_sql, hash, last_seen.
- `db_object_summaries` — object id, summary, business_purpose, risk_notes.
- `db_dependencies` — from_object, to_object, dependency_type.
- `business_glossary` — term, definition, synonyms, owning_team.
- `review_rules` — category, rule_text, severity_default, engine, examples.
- `review_history` — object id, review_date, issue_type, comment, resolved_flag.
- `object_embeddings` — object id, chunk id, embedding, content_hash.

Keeping embeddings in a separate table is aligned with pgvector guidance because vectors increase row width and are best queried independently, then joined back to metadata. With pgvector, you can index embeddings using HNSW for fast approximate search, and cosine distance is typically used for semantic retrieval; remember that the operator returns distance, not similarity, so you usually compute `1 - distance` if you want a similarity-style score. [postgresql](https://postgresql.us/events/pgconfnyc2024/sessions/session/1862/slides/172/pgvector_best_practices_pgconfnyc2024.pdf)

## Ingestion pipeline

Build an automated nightly or event-driven ingestion pipeline from your CRM repo and database metadata. Pull object definitions from `pg_proc`, `pg_views`, `information_schema`, migration folders, scheduler definitions, and any ETL repository that contains SQL used outside the database. Store a SHA256 hash for each object so only changed objects are re-summarized and re-embedded, which reduces embedding churn and keeps refresh costs low. [brandonwie](https://brandonwie.dev/posts/pgvector-hnsw-postgresql)

For each changed object:
- Extract raw SQL.
- Parse dependencies.
- Generate a concise technical summary.
- Generate a business-purpose summary.
- Chunk intelligently, for example by CTE, logical block, or procedure section.
- Embed summaries plus logical chunks.
- Update dependency graph and glossary links.

Do not chunk only by fixed token windows. SQL reviews benefit more from semantic chunks like “temp staging block,” “dedupe logic,” “final aggregation,” and “upsert into fact table,” because that maps to review reasoning better.

## How the reviewer should think

Your prompt should force a two-pass review:
1. **Technical pass** — syntax, performance, locking, indexing, null handling, transaction boundaries, anti-patterns.
2. **Business-logic pass** — Does the object implement CRM rules correctly based on glossary, dependencies, and adjacent objects?

Feed the model retrieved context in named sections like:
- `OBJECT_UNDER_REVIEW`
- `RELATED_OBJECTS`
- `BUSINESS_RULES`
- `KNOWN_INCIDENTS`
- `CHECKLIST_RULES`
- `SIMILAR_PAST_REVIEWS`

Then ask it for:
- correctness issues,
- business-rule mismatches,
- hidden assumptions,
- regression risks,
- missing tests,
- questionable naming or semantics,
- suggested SQL rewrites,
- queries/metrics to validate after deployment.

That structure reduces generic feedback and increases the chance the model reviews “real logic.”

## Add real value signals

To make the assistant genuinely useful, connect reviews to operational evidence. Include:
- slow query logs,
- top API endpoints calling each object,
- row counts and cardinality ranges,
- known lock/contention hotspots,
- failed job history,
- production incidents caused by related objects.

If the model sees that a view feeds a Streamlit dashboard refreshed every minute, its advice should differ from a nightly ETL staging proc. If it knows a table has 200M rows and heavy writes, it will judge indexes and joins differently. This kind of context is what turns an LLM from “SQL linter” into “database reviewer.”

## Guardrails

Never let the assistant run arbitrary SQL against production. Expose read-only, allowlisted tools only: object definition lookup, dependency graph query, sample row-count stats, index metadata, and maybe `EXPLAIN` in a safe non-prod clone. MCP-style tool exposure is useful here because it makes tool access explicit, typed, and auditable. [modelcontextprotocol](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

Also add:
- schema-level allowlists,
- prompt-injection resistance for retrieved text,
- clear labels for inferred vs verified claims,
- source citations inside the internal review UI,
- a “needs human validation” flag for uncertain business-rule judgments.

## Suggested implementation path

Start small and iterate:

| Phase | Goal | Deliverable |
|---|---|---|
| 1 | Build knowledge base | Postgres tables for objects, dependencies, glossary, rules  [brandonwie](https://brandonwie.dev/posts/pgvector-hnsw-postgresql) |
| 2 | Add retrieval | pgvector embeddings + lexical search + dependency expansion  [brandonwie](https://brandonwie.dev/posts/pgvector-hnsw-postgresql) |
| 3 | Add reviewer | Prompted local LLM that outputs severity-ranked findings |
| 4 | Add tool layer | Read-only metadata/MCP tools for live schema facts  [enterprisedb](https://www.enterprisedb.com/blog/building-real-time-data-aware-intelligence-postgres-and-model-context-protocol) |
| 5 | Add feedback loop | Store accepted/rejected comments and improve prompts/rules |
| 6 | Optional tuning | Fine-tune for review style, format, and prioritization |

For your stack, Phase 1–4 is enough to create a high-value internal reviewer without doing expensive model training.

## Practical stack for you

A strong local setup would be:
- **LLM**: Qwen-Coder / DeepSeek-Coder class model or another strong local coding model served via Ollama or vLLM.
- **Embeddings**: local embedding model.
- **Database**: PostgreSQL + pgvector.
- **Pipelines**: Python for ingestion and hashing.
- **API**: FastAPI review service.
- **UI**: Streamlit or internal web app.
- **Tool layer**: MCP-style read-only database tools for schema facts and dependency lookup. [ravikumarpidintla.medium](https://ravikumarpidintla.medium.com/bridging-mcp-with-databases-empowering-llms-for-incident-insights-poc-0cdaccbd0b25)

That matches your existing Python/Postgres/FastAPI strengths and minimizes new operational complexity.

## One concrete example

Suppose a developer changes `fn_monthly_distributor_sales`. A basic LLM might say: “Avoid `SELECT *`, consider indexing `sale_date`.” A grounded reviewer could say: “This function now aggregates invoice rows before return adjustment, but the glossary and related return-settlement function indicate returns must be netted at distributor-month grain before incentive calculation, otherwise incentive payouts will be overstated for partial-return months.” That second kind of review is only possible when the model has your CRM objects, glossary, and related logic in retrieval context.

## Recommendation

Build a **schema-aware CRM review copilot**, not a “trained-on-the-database” model first. Use RAG over SQL objects, business glossary, dependencies, incidents, and past review comments; back it with pgvector in Postgres; and expose safe metadata tools so the model can verify real facts before commenting. After that works, add lightweight fine-tuning only for review style and ranking consistency. [enterprisedb](https://www.enterprisedb.com/blog/building-real-time-data-aware-intelligence-postgres-and-model-context-protocol)

Would you like me to turn this into a concrete implementation blueprint next — including PostgreSQL schema, ingestion pipeline, embedding strategy, and a FastAPI + Streamlit architecture?

Follow-ups
Why architects disagree on whether RAG or Fine-Tuning is more reliable for enforcing complex business logic in code reviews
Yes, please provide the full blueprint
Show me the PostgreSQL schema for the knowledge base
How does the ingestion pipeline for CRM metadata work
Explain how MCP tools work for database lookups


-----------------
Here’s the full blueprint: build a **local CRM SQL Review Copilot** as a schema-aware RAG system with Postgres + pgvector, a local LLM, a read-only tool layer, and a review workflow that combines technical SQL checks with business-logic validation. This design keeps knowledge current, lets the assistant reason over real CRM objects and dependencies, and avoids the brittleness of retraining every time your schema changes. [brandonwie](https://brandonwie.dev/posts/pgvector-hnsw-postgresql)

## Target architecture

Use five services: **ingestion**, **knowledge store**, **retrieval**, **review engine**, and **UI/API**. PostgreSQL with pgvector works well as the core store because you can keep relational metadata, dependencies, and embeddings together, while HNSW indexes provide fast semantic retrieval; storing embeddings in a separate table is recommended to avoid bloating core object rows. [postgresql](https://postgresql.us/events/pgconfnyc2024/sessions/session/1862/slides/172/pgvector_best_practices_pgconfnyc2024.pdf)

A clean deployment for your stack is:
- PostgreSQL + pgvector for metadata, graph edges, and embeddings [brandonwie](https://brandonwie.dev/posts/pgvector-hnsw-postgresql)
- Python ingestion workers for parsing SQL objects and generating summaries [medium](https://medium.com/@fredyriveraacevedo13/building-a-fastapi-powered-rag-backend-with-postgresql-pgvector-c239f032508a)
- FastAPI as the orchestration API and review service, with typed request/response models and clean DB separation [medium](https://medium.com/@fredyriveraacevedo13/building-a-fastapi-powered-rag-backend-with-postgresql-pgvector-c239f032508a)
- Local LLM via Ollama or vLLM for generation
- Streamlit or a thin internal web app for review UX
- Optional MCP-style read-only tool server for safe schema lookups and metadata queries [uniclaw](https://uniclaw.ai/blog/ai-agent-database-sql-query)

## Data model

Design the store so the model can retrieve by object, dependency, business concept, and past incident. Separate embeddings from the main entity tables, and include a `content_hash` so unchanged objects are not re-embedded unnecessarily. [brandonwie](https://brandonwie.dev/ko/posts/pgvector-hnsw-postgresql)

Suggested schema:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE kb_db_objects (
    object_id           BIGSERIAL PRIMARY KEY,
    source_system       TEXT NOT NULL DEFAULT 'crm',
    db_name             TEXT NOT NULL,
    schema_name         TEXT NOT NULL,
    object_name         TEXT NOT NULL,
    object_type         TEXT NOT NULL, -- table, view, function, procedure, index, job, trigger
    object_signature    TEXT,
    object_fqn          TEXT NOT NULL UNIQUE,
    source_sql          TEXT NOT NULL,
    normalized_sql      TEXT,
    source_hash         TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'active',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

```sql
CREATE TABLE kb_object_docs (
    doc_id              BIGSERIAL PRIMARY KEY,
    object_id           BIGINT NOT NULL REFERENCES kb_db_objects(object_id) ON DELETE CASCADE,
    doc_kind            TEXT NOT NULL, -- raw_sql, summary, business_summary, chunk, review_note
    chunk_no            INT,
    title               TEXT,
    content             TEXT NOT NULL,
    content_hash        TEXT NOT NULL,
    token_estimate      INT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

```sql
CREATE TABLE kb_object_embeddings (
    embedding_id        BIGSERIAL PRIMARY KEY,
    doc_id              BIGINT NOT NULL REFERENCES kb_object_docs(doc_id) ON DELETE CASCADE,
    embedding_model     TEXT NOT NULL,
    embedding_dim       INT NOT NULL,
    embedding           vector(768) NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

```sql
CREATE INDEX idx_kb_object_embeddings_hnsw
ON kb_object_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

`<=>` in pgvector returns cosine **distance**, not similarity, so if you expose scores to users, convert with `1 - distance` for a similarity-style number. [brandonwie](https://brandonwie.dev/ko/posts/pgvector-hnsw-postgresql)

Add metadata tables:

```sql
CREATE TABLE kb_dependencies (
    dependency_id       BIGSERIAL PRIMARY KEY,
    from_object_id      BIGINT NOT NULL REFERENCES kb_db_objects(object_id) ON DELETE CASCADE,
    to_object_id        BIGINT NOT NULL REFERENCES kb_db_objects(object_id) ON DELETE CASCADE,
    dependency_type     TEXT NOT NULL, -- reads, writes, joins, calls, refreshes, derives_from
    confidence          NUMERIC(5,4),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

```sql
CREATE TABLE kb_business_glossary (
    glossary_id         BIGSERIAL PRIMARY KEY,
    domain_area         TEXT NOT NULL, -- crm, sales, collections, incentives
    term                TEXT NOT NULL,
    synonyms            TEXT[],
    definition          TEXT NOT NULL,
    business_rules      TEXT,
    owner_team          TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

```sql
CREATE TABLE kb_review_rules (
    rule_id             BIGSERIAL PRIMARY KEY,
    category            TEXT NOT NULL, -- performance, correctness, business_logic, security
    object_type         TEXT,          -- nullable => global
    rule_code           TEXT NOT NULL UNIQUE,
    severity_default    TEXT NOT NULL,
    rule_text           TEXT NOT NULL,
    rationale           TEXT,
    example_bad         TEXT,
    example_good        TEXT,
    engine              TEXT NOT NULL DEFAULT 'llm' -- llm, deterministic, hybrid
);
```

```sql
CREATE TABLE kb_incidents (
    incident_id         BIGSERIAL PRIMARY KEY,
    object_id           BIGINT REFERENCES kb_db_objects(object_id),
    incident_date       TIMESTAMPTZ NOT NULL,
    incident_type       TEXT NOT NULL, -- wrong totals, lock contention, timeout, deadlock
    summary             TEXT NOT NULL,
    impact              TEXT,
    root_cause          TEXT,
    mitigation          TEXT
);
```

```sql
CREATE TABLE kb_review_history (
    review_id           BIGSERIAL PRIMARY KEY,
    object_id           BIGINT REFERENCES kb_db_objects(object_id),
    review_scope        TEXT NOT NULL, -- PR, manual, scheduled
    model_name          TEXT NOT NULL,
    prompt_version      TEXT NOT NULL,
    findings_json       JSONB NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

For lexical retrieval, add `tsvector` columns or a dedicated search materialized view and combine that with vector search for hybrid retrieval, which is a common production pattern for exact identifiers plus semantic concepts. [ramnode](https://ramnode.com/guides/series/postgres-superstack/pgvector)

## Ingestion pipeline

Ingestion should be incremental and hash-based. Pull metadata from your repo, migrations, and live schema catalogs, then summarize and embed only changed artifacts; the `content_hash` pattern is specifically useful for skipping unnecessary re-embedding. [brandonwie](https://brandonwie.dev/posts/pgvector-hnsw-postgresql)

Recommended sources:
- Git repo: migrations, DDL scripts, ETL SQL, scheduler definitions
- Postgres catalogs: `pg_proc`, `pg_views`, `pg_trigger`, `pg_indexes`, `information_schema`
- Job scheduler metadata: Airflow DAG SQL references, cron jobs, pg_cron metadata
- App code references: FastAPI repositories or services that call SQL objects
- Operational evidence: slow query logs, deadlock events, failed ETL runs

Pipeline stages:
1. Discover objects.
2. Normalize SQL.
3. Parse dependencies.
4. Generate summaries.
5. Generate business summaries.
6. Chunk logically.
7. Embed summaries and chunks.
8. Upsert metadata, docs, embeddings, and edges.
9. Refresh lexical search artifacts.
10. Store lineage snapshots for diffing.

Useful normalized outputs per object:
- `raw_sql`
- `normalized_sql`
- `technical_summary`
- `business_summary`
- `dependency_summary`
- `review_context_summary`

A logical chunking strategy is better than fixed windows. Split by CTEs, temp-table phases, aggregation blocks, merge/upsert logic, exception handling blocks, and output sections, because those boundaries map to how reviewers reason about SQL.

## Metadata extraction

Dependency extraction is critical because value comes from seeing the surrounding landscape, not only the object under review. Parse tables referenced, write targets, join keys, called functions, temporary staging tables, and whether the object is upstream to dashboards, APIs, or ETL chains.

Store at least:
- referenced tables/views/functions
- read/write mode
- join columns
- key filters
- date grain used
- aggregation grain
- materialization pattern
- scheduler frequency
- downstream consumers
- owning business domain

For CRM logic, also tag business semantics:
- lead lifecycle
- distributor hierarchy
- sales hierarchy
- collections
- returns
- incentive eligibility
- monthly target grain
- order/invoice/payment semantic grain

These tags help the model identify issues like grain mismatch, duplicate counting, and incorrect netting logic.

## Embedding strategy

Use embeddings on **summaries and logical chunks**, not just raw SQL. Pure raw-SQL embeddings are weaker for business-rule retrieval because many important concepts are implied, not stated directly.

Recommended embedded content types:
- technical summary
- business summary
- glossary definitions
- incident summaries
- past accepted review comments
- logical code chunks

Retrieval flow:
- lexical pass on identifiers and exact object names
- vector pass on summaries/chunks
- dependency expansion on top-k objects
- reranking by object type match, domain match, and recency

HNSW is a strong default for pgvector approximate nearest-neighbor indexing, with `m = 16` and `ef_construction = 64` as widely used starting values; cosine distance is the standard choice for NLP-style embeddings. pgvector is a good fit when you already run Postgres and your use case benefits from relational joins and transactional metadata management. [postgresql](https://postgresql.us/events/pgconfnyc2024/sessions/session/1862/slides/172/pgvector_best_practices_pgconfnyc2024.pdf)

## Review engine

Use a multi-stage reviewer instead of one giant prompt. This improves grounding and makes it easier to debug.

Stage 1: **Object analyzer**
- identify object type
- extract touched entities
- infer purpose
- determine review profile

Stage 2: **Context retrieval**
- fetch relevant rules
- fetch glossary terms
- fetch incidents
- fetch related objects via dependencies
- fetch similar reviews

Stage 3: **Technical review**
- syntax and correctness
- null handling
- transaction behavior
- performance and index usage
- cardinality and join risk
- anti-patterns

Stage 4: **Business logic review**
- does the grain match the business requirement
- are returns/credits/netting rules respected
- are status filters aligned with glossary definitions
- are aggregations done at the right stage
- does naming reflect actual behavior

Stage 5: **Validation recommendations**
- SQL test cases
- sample assertions
- rollout checks
- metrics to monitor after deployment

Response schema:

```json
{
  "object_fqn": "crm.fn_monthly_distributor_sales",
  "overall_risk": "high",
  "findings": [
    {
      "severity": "critical",
      "category": "business_logic",
      "title": "Aggregation occurs before return netting",
      "evidence": "Related glossary and dependency context indicate returns must be netted at distributor-month grain.",
      "line_refs": ["42-58"],
      "impact": "Incentive overstatement for partial-return months",
      "recommendation": "Move return adjustment before monthly aggregation.",
      "confidence": 0.88
    }
  ],
  "validation_queries": [
    "compare monthly totals before/after with return-adjusted fact table",
    "check duplicate distributor-month rows after join"
  ],
  "deployment_checks": [
    "monitor monthly incentive delta > 1%",
    "monitor query runtime and temp spill"
  ]
}
```

## Prompt design

Use structured prompts with named context sections. The model should not guess missing facts if the retrieval did not supply them; it should explicitly mark uncertainty.

Prompt sections:
- `OBJECT_UNDER_REVIEW`
- `OBJECT_METADATA`
- `RELATED_OBJECTS`
- `BUSINESS_GLOSSARY`
- `INCIDENT_HISTORY`
- `CHECKLIST_RULES`
- `PAST_REVIEW_PATTERNS`
- `TASK_INSTRUCTIONS`
- `OUTPUT_SCHEMA`

Core instructions:
- separate verified facts from inference
- cite which retrieved object or rule supports each claim
- classify severity and risk type
- suggest minimal safe refactor first
- call out where live plan or row-count validation is still needed

This pattern helps the model review “real logic” rather than producing generic SQL commentary.

## Deterministic checks

Do not rely on the LLM for everything. Add deterministic validators for the easy, high-confidence cases:
- `SELECT *`
- missing schema qualification
- `OR` patterns on large tables
- function calls on indexed predicates
- broad `LEFT JOIN` with post-filter turning it into inner semantics
- missing `WHERE` in update/delete
- no exception handling in ETL wrapper functions
- materialized view refresh without dependency awareness
- missing partial index opportunity for status/date patterns

A hybrid approach gives you fast, explainable baseline findings and lets the LLM focus on logic and trade-offs.

## FastAPI service blueprint

A clean FastAPI structure is appropriate because it gives you typed APIs, automatic docs, and clean separation between orchestration and storage logic. [medium](https://medium.com/@fredyriveraacevedo13/building-a-fastapi-powered-rag-backend-with-postgresql-pgvector-c239f032508a)

Suggested layout:

```text
app/
  api/
    review.py
    ingest.py
    search.py
    admin.py
  core/
    config.py
    logging.py
    security.py
  db/
    session.py
    models.py
    repositories/
  services/
    ingestion_service.py
    parser_service.py
    embedding_service.py
    retrieval_service.py
    review_service.py
    rule_engine.py
    mcp_proxy_service.py
  schemas/
    review.py
    ingest.py
    search.py
  workers/
    ingest_worker.py
    summary_worker.py
    embedding_worker.py
```

Core endpoints:
- `POST /ingest/object`
- `POST /ingest/repo-sync`
- `POST /review/sql`
- `POST /review/pr-diff`
- `POST /review/object/{object_fqn}`
- `GET /search/context`
- `GET /object/{object_fqn}`
- `GET /dependencies/{object_fqn}`
- `GET /incidents/{object_fqn}`

FastAPI best practices like Pydantic models, environment-based configuration, parameterized SQL, and clean connection management are sensible here. [medium](https://medium.com/@fredyriveraacevedo13/building-a-fastapi-powered-rag-backend-with-postgresql-pgvector-c239f032508a)

## Streamlit or internal UI

A good first UI is a Streamlit app because it fits your current workflow and can be built quickly. Show four panels:
- object or PR input
- retrieved context preview
- findings with severity/risk tabs
- validation SQL / deployment checklist

Useful UX features:
- diff-aware review mode
- clickable dependency graph
- toggle between technical and business findings
- accepted/rejected feedback buttons
- links to source SQL and similar prior reviews

This feedback loop becomes your training data later.

## MCP / tool layer

If you add live tools, keep them strictly read-only. MCP-style tool exposure is useful because it standardizes external tool access for models and helps reduce hallucinated database facts by forcing the model to fetch approved metadata when needed. [enterprisedb](https://www.enterprisedb.com/blog/building-real-time-data-aware-intelligence-postgres-and-model-context-protocol)

Recommended read-only tools:
- `get_object_definition(object_fqn)`
- `get_object_dependencies(object_fqn, depth)`
- `get_table_stats(schema, table)`
- `get_index_metadata(schema, table)`
- `get_recent_incidents(object_fqn)`
- `search_glossary(term)`
- `explain_in_staging(sql)` only against non-prod with strict timeouts

Guardrails:
- dedicated read-only DB user
- schema allowlist
- query timeout
- row-limit cap
- no arbitrary SQL against production
- audit logging for every tool invocation

Read-only MCP/DB access is commonly recommended for safety when exposing databases to AI systems. [designrevision](https://designrevision.com/blog/supabase-mcp-server)

## Security model

Treat this as an internal developer platform, not a toy script. Use:
- separate DB roles for app, ingestion, and read-only tools
- no prod-write credentials anywhere in the review stack
- secrets via env or secret manager, not code [medium](https://medium.com/@fredyriveraacevedo13/building-a-fastapi-powered-rag-backend-with-postgresql-pgvector-c239f032508a)
- prompt-injection filters on retrieved content
- PII masking in indexed content
- row-level scoping if different teams should see different domains
- full review audit trail

If you ingest prod-derived logs, scrub customer identifiers before embedding.

## Deployment pattern

Start with one Postgres instance and one FastAPI app. pgvector is a good choice when you already operate Postgres and do not need a dedicated vector database footprint yet. Once usage grows, split: [brandonwie](https://brandonwie.dev/posts/pgvector-hnsw-postgresql)
- primary metadata DB
- ingestion workers
- review API
- LLM runtime
- optional reranker service

Suggested environments:
- local/dev on Docker Compose
- stage with sanitized metadata clone
- prod for metadata only, with staging explain tools

## Suggested Docker Compose

```yaml
version: "3.9"

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: crm_ai
      POSTGRES_USER: crm_ai
      POSTGRES_PASSWORD: crm_ai
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  api:
    build: ./api
    environment:
      DATABASE_URL: postgresql+psycopg://crm_ai:crm_ai@postgres:5432/crm_ai
      LLM_BASE_URL: http://ollama:11434
      EMBEDDING_MODEL: local-embed-model
      GENERATION_MODEL: local-review-model
    depends_on:
      - postgres
      - ollama
    ports:
      - "8000:8000"

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama:/root/.ollama

  streamlit:
    build: ./ui
    environment:
      API_BASE_URL: http://api:8000
    depends_on:
      - api
    ports:
      - "8501:8501"

volumes:
  pgdata:
  ollama:
```

## Retrieval SQL examples

Hybrid search example:

```sql
WITH lexical AS (
    SELECT d.doc_id, od.object_id, od.content,
           ts_rank_cd(to_tsvector('english', od.content), plainto_tsquery($1)) AS lexical_score
    FROM kb_object_docs od
    JOIN kb_db_objects o ON o.object_id = od.object_id
    JOIN kb_object_docs d ON d.doc_id = od.doc_id
    WHERE to_tsvector('english', od.content) @@ plainto_tsquery($1)
    LIMIT 20
),
semantic AS (
    SELECT od.doc_id, od.object_id, od.content,
           1 - (e.embedding <=> $2::vector) AS semantic_score
    FROM kb_object_docs od
    JOIN kb_object_embeddings e ON e.doc_id = od.doc_id
    ORDER BY e.embedding <=> $2::vector
    LIMIT 20
)
SELECT COALESCE(l.doc_id, s.doc_id) AS doc_id,
       COALESCE(l.object_id, s.object_id) AS object_id,
       COALESCE(l.content, s.content) AS content,
       COALESCE(l.lexical_score, 0) * 0.4 + COALESCE(s.semantic_score, 0) * 0.6 AS final_score
FROM lexical l
FULL OUTER JOIN semantic s ON l.doc_id = s.doc_id
ORDER BY final_score DESC
LIMIT 15;
```

The semantic score uses `1 - distance` because pgvector cosine returns distance, not similarity. [brandonwie](https://brandonwie.dev/ko/posts/pgvector-hnsw-postgresql)

## Review flow for a PR

When a SQL diff arrives:
1. detect changed object(s)
2. parse touched tables/columns
3. fetch same object previous version
4. retrieve top related docs and glossary entries
5. traverse dependencies one or two hops
6. run deterministic rules
7. run LLM technical review
8. run LLM business-logic review
9. merge findings
10. store result and reviewer feedback

This should take a few seconds locally if retrieval is tight and prompts are constrained.

## Feedback loop

Capture reviewer interactions from day one. Store:
- accepted findings
- dismissed findings
- edited findings
- missing finding reports
- severity changes by human reviewer

That data becomes:
- prompt improvement material
- reranker training data
- optional fine-tuning data later

Fine-tuning is most useful once you have enough accepted review examples to teach tone, prioritization, and organization, while current schema knowledge still comes from RAG. [arxiv](https://arxiv.org/abs/2511.05302)

## Example rollout plan

### Phase 1
Ingest only:
- tables
- views
- functions
- glossary
- rules

Deliver:
- search UI
- object summary
- manual review endpoint

### Phase 2
Add:
- dependency graph
- incidents
- PR diff review
- deterministic rule engine

Deliver:
- severity-ranked reviews
- validation queries
- monitoring checklist

### Phase 3
Add:
- app-to-DB usage mapping from FastAPI
- Streamlit/dashboard lineage
- ETL job context
- read-only live tools / MCP server

Deliver:
- context-rich business-logic review
- deployment-aware suggestions

### Phase 4
Add:
- accepted-review feedback loop
- reranking
- optional fine-tuning for style and consistency

## What “good” looks like

A mature review should say things like:
- “This function is technically valid, but the aggregation grain changed from invoice-month to distributor-month without preserving return adjustments, which conflicts with the glossary definition of net sales.”
- “This partial index recommendation helps the API path, but it may hurt high-volume ETL writes; consider a reporting-specific materialized view instead.”
- “This dashboard view is correct for Streamlit reads, but it duplicates logic already implemented in the ETL summary proc, which increases semantic drift risk.”

Those are high-value comments because they connect SQL, system behavior, and business logic.

## Recommendation for your stack

Given your background, the best implementation path is:
- PostgreSQL + pgvector as the knowledge backbone [postgresql](https://postgresql.us/events/pgconfnyc2024/sessions/session/1862/slides/172/pgvector_best_practices_pgconfnyc2024.pdf)
- Python ingestion and retrieval services
- FastAPI orchestration and review API [medium](https://medium.com/@fredyriveraacevedo13/building-a-fastapi-powered-rag-backend-with-postgresql-pgvector-c239f032508a)
- Streamlit reviewer console
- local LLM via Ollama/vLLM
- optional MCP-style read-only tools for schema-safe live context [modelcontextprotocol](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

That gets you a production-capable internal reviewer without overcomplicating the first version.

Would you like the next step to be:
1. full PostgreSQL DDL for all tables,
2. FastAPI project skeleton,
3. ingestion pipeline code, or
4. the complete review prompt pack?

-------------------------------
Yes — here are the four options, one by one, exactly as requested:

1. Full PostgreSQL DDL for all tables  
2. FastAPI project skeleton  
3. Ingestion pipeline code  
4. Complete review prompt pack  

Best order is usually:
- Start with **1. Full PostgreSQL DDL** because it defines the backbone of the knowledge base and constraints. [postgresql](https://www.postgresql.org/docs/current/sql-createtable.html)
- Then **2. FastAPI project skeleton** so the app structure stays clean and layered from the beginning. [medium](https://medium.com/@khalil.saidane/scalabel-fastapi-project-layered-architecture-10852a40fd38)
- Then **3. Ingestion pipeline code** because it depends on the schema and service boundaries already being defined. [medium](https://medium.com/@fredyriveraacevedo13/building-a-fastapi-powered-rag-backend-with-postgresql-pgvector-c239f032508a)
- Then **4. Complete review prompt pack** so prompts match the actual schema, retrieval model, and API workflow. [medium](https://medium.com/@khalil.saidane/scalabel-fastapi-project-layered-architecture-10852a40fd38)

Send:
- **1** for full PostgreSQL DDL,
- **2** for FastAPI skeleton,
- **3** for ingestion pipeline,
- **4** for prompt pack.

- -------------------------
Below is a production-style **full PostgreSQL DDL** for the CRM SQL Review Copilot knowledge base. It is designed for PostgreSQL with pgvector, uses explicit primary/foreign keys and constraints, and keeps embeddings in a separate table so vector storage does not bloat the main metadata rows. [postgresql](https://www.postgresql.org/docs/current/sql-createtable.html)

## Scope

This schema covers:
- database objects and versions,
- logical document chunks and embeddings,
- dependencies and lineage,
- business glossary and rules,
- incidents, reviews, feedback, and ingestion runs.

It also includes practical indexes, checks, audit timestamps, and a small amount of JSONB only where flexible payloads are useful; JSONB is powerful, but should be used selectively because large semi-structured blobs can create performance and storage trade-offs if overused. [sitepoint](https://www.sitepoint.com/postgresql-jsonb-query-performance-indexing/)

## Full DDL

```sql
BEGIN;

CREATE SCHEMA IF NOT EXISTS ai_kb;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

SET search_path TO ai_kb, public;

-- =========================================================
-- 1. Common functions
-- =========================================================

CREATE OR REPLACE FUNCTION ai_kb.set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

-- =========================================================
-- 2. Enumerated domains via CHECK constraints
-- =========================================================

CREATE TABLE ai_kb.ref_object_type (
    object_type TEXT PRIMARY KEY,
    description TEXT NOT NULL
);

INSERT INTO ai_kb.ref_object_type (object_type, description) VALUES
('table', 'Base table'),
('view', 'Logical view'),
('materialized_view', 'Materialized view'),
('function', 'PostgreSQL function'),
('procedure', 'Stored procedure'),
('index', 'Index definition'),
('trigger', 'Trigger definition'),
('job', 'Scheduled job'),
('script', 'Standalone SQL script'),
('query_template', 'Application-side SQL text template');

CREATE TABLE ai_kb.ref_doc_kind (
    doc_kind TEXT PRIMARY KEY,
    description TEXT NOT NULL
);

INSERT INTO ai_kb.ref_doc_kind (doc_kind, description) VALUES
('raw_sql', 'Raw source SQL'),
('normalized_sql', 'Normalized SQL'),
('technical_summary', 'Technical summary'),
('business_summary', 'Business summary'),
('logic_chunk', 'Logical code chunk'),
('review_note', 'Stored review note'),
('incident_note', 'Incident summary'),
('glossary_entry', 'Business glossary content'),
('rule_text', 'Review rule text'),
('test_case', 'Suggested test case'),
('runbook_note', 'Operational or deployment note');

CREATE TABLE ai_kb.ref_dependency_type (
    dependency_type TEXT PRIMARY KEY,
    description TEXT NOT NULL
);

INSERT INTO ai_kb.ref_dependency_type (dependency_type, description) VALUES
('reads', 'Reads from object'),
('writes', 'Writes to object'),
('calls', 'Calls procedure/function'),
('joins', 'Joins to object'),
('refreshes', 'Refreshes materialized object'),
('depends_on', 'General dependency'),
('derived_from', 'Derived from upstream object'),
('feeds', 'Feeds downstream object'),
('used_by_api', 'Used by API/service'),
('used_by_dashboard', 'Used by dashboard/report'),
('used_by_etl', 'Used by ETL job');

CREATE TABLE ai_kb.ref_severity (
    severity TEXT PRIMARY KEY,
    severity_rank SMALLINT NOT NULL UNIQUE CHECK (severity_rank > 0)
);

INSERT INTO ai_kb.ref_severity (severity, severity_rank) VALUES
('critical', 1),
('high', 2),
('medium', 3),
('low', 4),
('info', 5);

CREATE TABLE ai_kb.ref_rule_category (
    category TEXT PRIMARY KEY,
    description TEXT NOT NULL
);

INSERT INTO ai_kb.ref_rule_category (category, description) VALUES
('performance', 'Execution efficiency and indexing'),
('correctness', 'Data correctness and semantic correctness'),
('business_logic', 'Domain logic and business-rule alignment'),
('security', 'Security, permissions, exposure, injection'),
('maintainability', 'Naming, structure, readability, reuse'),
('operability', 'Monitoring, deployment, rollback, runtime behavior');

CREATE TABLE ai_kb.ref_review_scope (
    review_scope TEXT PRIMARY KEY,
    description TEXT NOT NULL
);

INSERT INTO ai_kb.ref_review_scope (review_scope, description) VALUES
('manual', 'Manual one-off review'),
('pull_request', 'Review against a code diff'),
('scheduled', 'Periodic background review'),
('pre_deploy', 'Pre-release review'),
('post_incident', 'Review triggered by incident'),
('regression_check', 'Focused review for regression');

CREATE TABLE ai_kb.ref_feedback_label (
    feedback_label TEXT PRIMARY KEY,
    description TEXT NOT NULL
);

INSERT INTO ai_kb.ref_feedback_label (feedback_label, description) VALUES
('accepted', 'Reviewer accepted the finding'),
('rejected', 'Reviewer rejected the finding'),
('edited', 'Reviewer edited the finding'),
('missing_issue', 'Assistant missed an issue'),
('false_positive', 'Assistant incorrectly flagged an issue');

-- =========================================================
-- 3. Source systems and ingestion runs
-- =========================================================

CREATE TABLE ai_kb.source_system (
    source_system_id BIGSERIAL PRIMARY KEY,
    source_code TEXT NOT NULL UNIQUE,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('database', 'git_repo', 'scheduler', 'app_code', 'log_source')),
    connection_hint TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER trg_source_system_updated_at
BEFORE UPDATE ON ai_kb.source_system
FOR EACH ROW
EXECUTE FUNCTION ai_kb.set_updated_at();

CREATE TABLE ai_kb.ingestion_run (
    ingestion_run_id BIGSERIAL PRIMARY KEY,
    source_system_id BIGINT NOT NULL REFERENCES ai_kb.source_system(source_system_id),
    run_type TEXT NOT NULL CHECK (run_type IN ('full', 'incremental', 'rebuild_embeddings', 'rebuild_summaries', 'backfill')),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'partial')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    objects_scanned INT NOT NULL DEFAULT 0 CHECK (objects_scanned >= 0),
    objects_changed INT NOT NULL DEFAULT 0 CHECK (objects_changed >= 0),
    docs_embedded INT NOT NULL DEFAULT 0 CHECK (docs_embedded >= 0),
    error_count INT NOT NULL DEFAULT 0 CHECK (error_count >= 0),
    notes TEXT,
    metrics_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_ingestion_run_source_started
ON ai_kb.ingestion_run (source_system_id, started_at DESC);

-- =========================================================
-- 4. Core objects
-- =========================================================

CREATE TABLE ai_kb.db_object (
    object_id BIGSERIAL PRIMARY KEY,
    source_system_id BIGINT NOT NULL REFERENCES ai_kb.source_system(source_system_id),
    db_name TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    object_name TEXT NOT NULL,
    object_type TEXT NOT NULL REFERENCES ai_kb.ref_object_type(object_type),
    object_signature TEXT,
    object_fqn TEXT NOT NULL,
    owner_name TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_db_object_fqn UNIQUE (source_system_id, object_fqn),
    CONSTRAINT ck_db_object_name_nonempty CHECK (length(trim(object_name)) > 0),
    CONSTRAINT ck_db_object_schema_nonempty CHECK (length(trim(schema_name)) > 0)
);

CREATE INDEX idx_db_object_lookup
ON ai_kb.db_object (schema_name, object_name, object_type);

CREATE INDEX idx_db_object_fqn_trgm
ON ai_kb.db_object
USING gin (object_fqn gin_trgm_ops);

CREATE TRIGGER trg_db_object_updated_at
BEFORE UPDATE ON ai_kb.db_object
FOR EACH ROW
EXECUTE FUNCTION ai_kb.set_updated_at();

CREATE TABLE ai_kb.db_object_version (
    object_version_id BIGSERIAL PRIMARY KEY,
    object_id BIGINT NOT NULL REFERENCES ai_kb.db_object(object_id) ON DELETE CASCADE,
    version_no INT NOT NULL CHECK (version_no > 0),
    source_hash TEXT NOT NULL,
    source_sql TEXT NOT NULL,
    normalized_sql TEXT,
    parser_version TEXT,
    extracted_metadata_json JSONB,
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to TIMESTAMPTZ,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_object_version UNIQUE (object_id, version_no),
    CONSTRAINT uq_object_current_version UNIQUE (object_id, is_current) DEFERRABLE INITIALLY IMMEDIATE
);

CREATE INDEX idx_db_object_version_object_current
ON ai_kb.db_object_version (object_id, is_current DESC, version_no DESC);

CREATE INDEX idx_db_object_version_hash
ON ai_kb.db_object_version (source_hash);

-- Note:
-- uq_object_current_version allows only one TRUE and one FALSE if used literally.
-- So we additionally enforce the intended behavior with a partial unique index below and keep is_current flexible.

ALTER TABLE ai_kb.db_object_version
DROP CONSTRAINT uq_object_current_version;

CREATE UNIQUE INDEX uq_db_object_version_current_true
ON ai_kb.db_object_version (object_id)
WHERE is_current;

CREATE TABLE ai_kb.object_tag (
    object_tag_id BIGSERIAL PRIMARY KEY,
    object_id BIGINT NOT NULL REFERENCES ai_kb.db_object(object_id) ON DELETE CASCADE,
    tag_key TEXT NOT NULL,
    tag_value TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_object_tag UNIQUE (object_id, tag_key, tag_value)
);

CREATE INDEX idx_object_tag_key_value
ON ai_kb.object_tag (tag_key, tag_value);

-- =========================================================
-- 5. Documents and embeddings
-- =========================================================

CREATE TABLE ai_kb.object_doc (
    doc_id BIGSERIAL PRIMARY KEY,
    object_id BIGINT NOT NULL REFERENCES ai_kb.db_object(object_id) ON DELETE CASCADE,
    object_version_id BIGINT REFERENCES ai_kb.db_object_version(object_version_id) ON DELETE CASCADE,
    doc_kind TEXT NOT NULL REFERENCES ai_kb.ref_doc_kind(doc_kind),
    chunk_no INT CHECK (chunk_no IS NULL OR chunk_no >= 0),
    title TEXT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    token_estimate INT CHECK (token_estimate IS NULL OR token_estimate >= 0),
    language_code TEXT DEFAULT 'sql',
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_object_doc_hash UNIQUE (object_id, doc_kind, content_hash, chunk_no)
);

CREATE INDEX idx_object_doc_object_kind
ON ai_kb.object_doc (object_id, doc_kind, created_at DESC);

CREATE INDEX idx_object_doc_version
ON ai_kb.object_doc (object_version_id);

CREATE INDEX idx_object_doc_content_trgm
ON ai_kb.object_doc
USING gin (content gin_trgm_ops);

CREATE TABLE ai_kb.embedding_model (
    embedding_model_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL UNIQUE,
    dimensions INT NOT NULL CHECK (dimensions > 0),
    distance_metric TEXT NOT NULL CHECK (distance_metric IN ('cosine', 'l2', 'inner_product')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE ai_kb.object_doc_embedding (
    embedding_id BIGSERIAL PRIMARY KEY,
    doc_id BIGINT NOT NULL REFERENCES ai_kb.object_doc(doc_id) ON DELETE CASCADE,
    embedding_model_id BIGINT NOT NULL REFERENCES ai_kb.embedding_model(embedding_model_id),
    content_hash TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_object_doc_embedding UNIQUE (doc_id, embedding_model_id, content_hash)
);

CREATE INDEX idx_object_doc_embedding_doc
ON ai_kb.object_doc_embedding (doc_id, embedding_model_id);

CREATE INDEX idx_object_doc_embedding_hnsw
ON ai_kb.object_doc_embedding
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- If you change embedding dimensions, use one table per embedding dimension
-- or standardize on one embedding model for the first release.

-- =========================================================
-- 6. Dependencies and lineage
-- =========================================================

CREATE TABLE ai_kb.object_dependency (
    dependency_id BIGSERIAL PRIMARY KEY,
    from_object_id BIGINT NOT NULL REFERENCES ai_kb.db_object(object_id) ON DELETE CASCADE,
    to_object_id BIGINT NOT NULL REFERENCES ai_kb.db_object(object_id) ON DELETE CASCADE,
    dependency_type TEXT NOT NULL REFERENCES ai_kb.ref_dependency_type(dependency_type),
    dependency_detail TEXT,
    join_columns TEXT[],
    confidence NUMERIC(5,4) CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    discovered_by TEXT NOT NULL CHECK (discovered_by IN ('parser', 'catalog', 'manual', 'llm')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_dependency_not_self CHECK (from_object_id <> to_object_id),
    CONSTRAINT uq_object_dependency UNIQUE (from_object_id, to_object_id, dependency_type, discovered_by)
);

CREATE INDEX idx_object_dependency_from
ON ai_kb.object_dependency (from_object_id, dependency_type);

CREATE INDEX idx_object_dependency_to
ON ai_kb.object_dependency (to_object_id, dependency_type);

CREATE TABLE ai_kb.object_usage_context (
    usage_context_id BIGSERIAL PRIMARY KEY,
    object_id BIGINT NOT NULL REFERENCES ai_kb.db_object(object_id) ON DELETE CASCADE,
    consumer_type TEXT NOT NULL CHECK (consumer_type IN ('fastapi', 'streamlit', 'etl', 'scheduler', 'report', 'external_app')),
    consumer_name TEXT NOT NULL,
    call_pattern TEXT,
    frequency_hint TEXT,
    latency_slo_ms INT CHECK (latency_slo_ms IS NULL OR latency_slo_ms > 0),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_object_usage_context UNIQUE (object_id, consumer_type, consumer_name)
);

CREATE INDEX idx_object_usage_context_object
ON ai_kb.object_usage_context (object_id, consumer_type);

-- =========================================================
-- 7. Business glossary
-- =========================================================

CREATE TABLE ai_kb.business_glossary (
    glossary_id BIGSERIAL PRIMARY KEY,
    domain_area TEXT NOT NULL,
    term TEXT NOT NULL,
    canonical_term TEXT NOT NULL,
    definition TEXT NOT NULL,
    business_rules TEXT,
    grain_hint TEXT,
    owner_team TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_business_glossary_term UNIQUE (domain_area, canonical_term)
);

CREATE INDEX idx_business_glossary_term_trgm
ON ai_kb.business_glossary
USING gin (canonical_term gin_trgm_ops);

CREATE TRIGGER trg_business_glossary_updated_at
BEFORE UPDATE ON ai_kb.business_glossary
FOR EACH ROW
EXECUTE FUNCTION ai_kb.set_updated_at();

CREATE TABLE ai_kb.business_glossary_synonym (
    glossary_synonym_id BIGSERIAL PRIMARY KEY,
    glossary_id BIGINT NOT NULL REFERENCES ai_kb.business_glossary(glossary_id) ON DELETE CASCADE,
    synonym TEXT NOT NULL,
    CONSTRAINT uq_business_glossary_synonym UNIQUE (glossary_id, synonym)
);

CREATE TABLE ai_kb.object_glossary_link (
    object_glossary_link_id BIGSERIAL PRIMARY KEY,
    object_id BIGINT NOT NULL REFERENCES ai_kb.db_object(object_id) ON DELETE CASCADE,
    glossary_id BIGINT NOT NULL REFERENCES ai_kb.business_glossary(glossary_id) ON DELETE CASCADE,
    link_type TEXT NOT NULL CHECK (link_type IN ('implements', 'references', 'aggregates', 'derives', 'violates_risk')),
    confidence NUMERIC(5,4) CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_object_glossary_link UNIQUE (object_id, glossary_id, link_type)
);

CREATE INDEX idx_object_glossary_link_object
ON ai_kb.object_glossary_link (object_id);

-- =========================================================
-- 8. Rules and prompt assets
-- =========================================================

CREATE TABLE ai_kb.review_rule (
    rule_id BIGSERIAL PRIMARY KEY,
    rule_code TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL REFERENCES ai_kb.ref_rule_category(category),
    object_type TEXT REFERENCES ai_kb.ref_object_type(object_type),
    severity_default TEXT NOT NULL REFERENCES ai_kb.ref_severity(severity),
    engine TEXT NOT NULL CHECK (engine IN ('deterministic', 'llm', 'hybrid')),
    title TEXT NOT NULL,
    rule_text TEXT NOT NULL,
    rationale TEXT,
    example_bad TEXT,
    example_good TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_review_rule_category_object
ON ai_kb.review_rule (category, object_type, is_active);

CREATE TRIGGER trg_review_rule_updated_at
BEFORE UPDATE ON ai_kb.review_rule
FOR EACH ROW
EXECUTE FUNCTION ai_kb.set_updated_at();

CREATE TABLE ai_kb.prompt_template (
    prompt_template_id BIGSERIAL PRIMARY KEY,
    template_code TEXT NOT NULL UNIQUE,
    template_name TEXT NOT NULL,
    template_scope TEXT NOT NULL CHECK (template_scope IN ('review', 'summary', 'retrieval', 'classification', 'rerank')),
    object_type TEXT REFERENCES ai_kb.ref_object_type(object_type),
    version_no INT NOT NULL CHECK (version_no > 0),
    template_text TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_prompt_template_version UNIQUE (template_code, version_no)
);

CREATE INDEX idx_prompt_template_scope
ON ai_kb.prompt_template (template_scope, object_type, is_active);

-- =========================================================
-- 9. Incidents and operational evidence
-- =========================================================

CREATE TABLE ai_kb.incident (
    incident_id BIGSERIAL PRIMARY KEY,
    object_id BIGINT REFERENCES ai_kb.db_object(object_id) ON DELETE SET NULL,
    incident_key TEXT UNIQUE,
    incident_date TIMESTAMPTZ NOT NULL,
    incident_type TEXT NOT NULL CHECK (incident_type IN ('wrong_total', 'timeout', 'lock_contention', 'deadlock', 'duplication', 'data_loss', 'stale_dashboard', 'etl_failure', 'security_exposure', 'other')),
    severity TEXT NOT NULL REFERENCES ai_kb.ref_severity(severity),
    summary TEXT NOT NULL,
    impact TEXT,
    root_cause TEXT,
    mitigation TEXT,
    metadata_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_incident_object_date
ON ai_kb.incident (object_id, incident_date DESC);

CREATE TABLE ai_kb.object_runtime_stat (
    runtime_stat_id BIGSERIAL PRIMARY KEY,
    object_id BIGINT NOT NULL REFERENCES ai_kb.db_object(object_id) ON DELETE CASCADE,
    stat_date DATE NOT NULL,
    execution_count BIGINT CHECK (execution_count IS NULL OR execution_count >= 0),
    mean_latency_ms NUMERIC(18,4) CHECK (mean_latency_ms IS NULL OR mean_latency_ms >= 0),
    p95_latency_ms NUMERIC(18,4) CHECK (p95_latency_ms IS NULL OR p95_latency_ms >= 0),
    rows_read BIGINT CHECK (rows_read IS NULL OR rows_read >= 0),
    rows_written BIGINT CHECK (rows_written IS NULL OR rows_written >= 0),
    temp_spill_mb NUMERIC(18,4) CHECK (temp_spill_mb IS NULL OR temp_spill_mb >= 0),
    error_count BIGINT CHECK (error_count IS NULL OR error_count >= 0),
    source_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_object_runtime_stat UNIQUE (object_id, stat_date, source_name)
);

CREATE INDEX idx_object_runtime_stat_object_date
ON ai_kb.object_runtime_stat (object_id, stat_date DESC);

-- =========================================================
-- 10. Reviews and findings
-- =========================================================

CREATE TABLE ai_kb.review_run (
    review_run_id BIGSERIAL PRIMARY KEY,
    review_scope TEXT NOT NULL REFERENCES ai_kb.ref_review_scope(review_scope),
    object_id BIGINT REFERENCES ai_kb.db_object(object_id) ON DELETE SET NULL,
    object_version_id BIGINT REFERENCES ai_kb.db_object_version(object_version_id) ON DELETE SET NULL,
    llm_model_name TEXT NOT NULL,
    embedding_model_name TEXT,
    prompt_template_id BIGINT REFERENCES ai_kb.prompt_template(prompt_template_id),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'partial')),
    overall_risk TEXT REFERENCES ai_kb.ref_severity(severity),
    input_payload_json JSONB,
    retrieved_context_json JSONB,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_review_run_object_date
ON ai_kb.review_run (object_id, started_at DESC);

CREATE TABLE ai_kb.review_finding (
    finding_id BIGSERIAL PRIMARY KEY,
    review_run_id BIGINT NOT NULL REFERENCES ai_kb.review_run(review_run_id) ON DELETE CASCADE,
    rule_id BIGINT REFERENCES ai_kb.review_rule(rule_id) ON DELETE SET NULL,
    category TEXT NOT NULL REFERENCES ai_kb.ref_rule_category(category),
    severity TEXT NOT NULL REFERENCES ai_kb.ref_severity(severity),
    title TEXT NOT NULL,
    evidence TEXT,
    impact TEXT,
    recommendation TEXT,
    risk_type TEXT,
    confidence NUMERIC(5,4) CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    line_start INT CHECK (line_start IS NULL OR line_start > 0),
    line_end INT CHECK (line_end IS NULL OR line_end > 0),
    source_kind TEXT NOT NULL CHECK (source_kind IN ('deterministic', 'llm', 'hybrid')),
    is_business_logic BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_review_finding_line_range CHECK (
        (line_start IS NULL AND line_end IS NULL)
        OR (line_start IS NOT NULL AND line_end IS NOT NULL AND line_end >= line_start)
    )
);

CREATE INDEX idx_review_finding_review_severity
ON ai_kb.review_finding (review_run_id, severity);

CREATE INDEX idx_review_finding_rule
ON ai_kb.review_finding (rule_id);

CREATE TABLE ai_kb.review_validation_query (
    validation_query_id BIGSERIAL PRIMARY KEY,
    review_run_id BIGINT NOT NULL REFERENCES ai_kb.review_run(review_run_id) ON DELETE CASCADE,
    query_order INT NOT NULL CHECK (query_order > 0),
    title TEXT NOT NULL,
    sql_text TEXT NOT NULL,
    purpose TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_review_validation_query UNIQUE (review_run_id, query_order)
);

CREATE TABLE ai_kb.review_deployment_check (
    deployment_check_id BIGSERIAL PRIMARY KEY,
    review_run_id BIGINT NOT NULL REFERENCES ai_kb.review_run(review_run_id) ON DELETE CASCADE,
    check_order INT NOT NULL CHECK (check_order > 0),
    check_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_review_deployment_check UNIQUE (review_run_id, check_order)
);

-- =========================================================
-- 11. Human feedback loop
-- =========================================================

CREATE TABLE ai_kb.review_feedback (
    feedback_id BIGSERIAL PRIMARY KEY,
    review_run_id BIGINT NOT NULL REFERENCES ai_kb.review_run(review_run_id) ON DELETE CASCADE,
    finding_id BIGINT REFERENCES ai_kb.review_finding(finding_id) ON DELETE CASCADE,
    feedback_label TEXT NOT NULL REFERENCES ai_kb.ref_feedback_label(feedback_label),
    reviewer_name TEXT,
    comments TEXT,
    edited_finding_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_review_feedback_review
ON ai_kb.review_feedback (review_run_id, created_at DESC);

-- =========================================================
-- 12. Optional retrieval logging
-- =========================================================

CREATE TABLE ai_kb.retrieval_log (
    retrieval_log_id BIGSERIAL PRIMARY KEY,
    review_run_id BIGINT REFERENCES ai_kb.review_run(review_run_id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    lexical_hits INT CHECK (lexical_hits IS NULL OR lexical_hits >= 0),
    semantic_hits INT CHECK (semantic_hits IS NULL OR semantic_hits >= 0),
    topk_used INT CHECK (topk_used IS NULL OR topk_used > 0),
    retrieval_payload_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_retrieval_log_review
ON ai_kb.retrieval_log (review_run_id, created_at DESC);

-- =========================================================
-- 13. Helpful views
-- =========================================================

CREATE OR REPLACE VIEW ai_kb.v_current_object_version AS
SELECT
    o.object_id,
    o.object_fqn,
    o.object_type,
    v.object_version_id,
    v.version_no,
    v.source_hash,
    v.source_sql,
    v.normalized_sql,
    v.created_at AS version_created_at
FROM ai_kb.db_object o
JOIN ai_kb.db_object_version v
  ON v.object_id = o.object_id
 AND v.is_current = TRUE
WHERE o.is_active = TRUE;

CREATE OR REPLACE VIEW ai_kb.v_object_doc_search AS
SELECT
    d.doc_id,
    d.object_id,
    o.object_fqn,
    o.object_type,
    d.doc_kind,
    d.title,
    d.content,
    to_tsvector('english', coalesce(d.title, '') || ' ' || d.content) AS search_vector
FROM ai_kb.object_doc d
JOIN ai_kb.db_object o
  ON o.object_id = d.object_id;

-- =========================================================
-- 14. Seed recommendations (empty inserts left to app/migrations)
-- =========================================================
-- Populate ai_kb.embedding_model after deciding on a single embedding model.
-- Populate ai_kb.review_rule and ai_kb.prompt_template from application migrations or seed scripts.

COMMIT;
```

## Design notes

This schema keeps **objects**, **versions**, **docs**, and **embeddings** separate so you can independently version SQL text, summaries, and vector representations without rewriting the whole object record every time. The HNSW index on the vector column is aligned with pgvector guidance for approximate nearest-neighbor search, and cosine distance is the right default for most text-embedding retrieval use cases. [cloud.ibm](https://cloud.ibm.com/docs/databases-for-postgresql?topic=databases-for-postgresql-pgvector)

A few implementation choices are deliberate:
- `db_object_version` lets you compare current vs prior versions cleanly.
- `object_doc` supports multiple representations per object: raw SQL, summary, chunk, incident note, etc.
- `object_doc_embedding` is split out so vector storage stays isolated from OLTP-style metadata queries, which is a common production recommendation for pgvector-backed systems. [brandonwie](https://brandonwie.dev/posts/pgvector-hnsw-postgresql)
- JSONB is used only for flexible metadata payloads, not as a substitute for normalized relational columns, because uncontrolled JSONB growth can make indexing and storage behavior harder to manage at scale. [aws.amazon](https://aws.amazon.com/blogs/database/postgresql-as-a-json-database-advanced-patterns-and-best-practices/)

## Recommended next steps

After creating the schema, do these next:

1. Seed `embedding_model` with your chosen embedding model and dimension.
2. Add migration scripts instead of manually editing the schema in production, since PostgreSQL schema evolution is safer and more auditable that way. [postgresql](https://www.postgresql.org/docs/current/sql-createtable.html)
3. Build ingestion in this order:
   - source systems,
   - db objects,
   - object versions,
   - docs,
   - embeddings,
   - dependencies,
   - glossary links.
4. Add a hybrid retrieval query using trigram/tsvector + pgvector, because exact object-name matches and semantic matches complement each other well. [ramnode](https://ramnode.com/guides/series/postgres-superstack/pgvector)

The next item in your sequence is **2. FastAPI project skeleton**.

--------------

Below is a **production-ready FastAPI project skeleton** for your CRM SQL Review Copilot. It follows the common FastAPI guidance of thin routers, explicit service/repository layers, typed schemas, and clean separation between API, application logic, and infrastructure, which helps maintainability as the system grows. [zestminds](https://www.zestminds.com/blog/fastapi-project-structure/)

## Structure

This layout keeps FastAPI-specific concerns near the edges, while review logic, retrieval, and ingestion stay in services and repositories. That separation aligns with layered FastAPI architecture guidance and makes testing easier because routers stay thin and dependencies are injected rather than hardcoded. [zyneto](https://zyneto.com/blog/best-practices-in-fastapi-architecture)

```text
crm-sql-review-copilot/
├── app/
│   ├── main.py
│   ├── lifespan.py
│   ├── api/
│   │   ├── deps.py
│   │   ├── errors.py
│   │   └── v1/
│   │       ├── api.py
│   │       └── endpoints/
│   │           ├── health.py
│   │           ├── objects.py
│   │           ├── search.py
│   │           ├── ingest.py
│   │           ├── review.py
│   │           ├── glossary.py
│   │           └── feedback.py
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── security.py
│   │   └── constants.py
│   ├── db/
│   │   ├── base.py
│   │   ├── session.py
│   │   ├── models/
│   │   │   ├── source_system.py
│   │   │   ├── db_object.py
│   │   │   ├── db_object_version.py
│   │   │   ├── object_doc.py
│   │   │   ├── object_doc_embedding.py
│   │   │   ├── object_dependency.py
│   │   │   ├── business_glossary.py
│   │   │   ├── review_rule.py
│   │   │   ├── review_run.py
│   │   │   ├── review_finding.py
│   │   │   ├── review_feedback.py
│   │   │   ├── incident.py
│   │   │   └── ingestion_run.py
│   │   └── repositories/
│   │       ├── source_system_repo.py
│   │       ├── object_repo.py
│   │       ├── object_version_repo.py
│   │       ├── object_doc_repo.py
│   │       ├── embedding_repo.py
│   │       ├── dependency_repo.py
│   │       ├── glossary_repo.py
│   │       ├── rule_repo.py
│   │       ├── review_repo.py
│   │       ├── feedback_repo.py
│   │       └── incident_repo.py
│   ├── schemas/
│   │   ├── common.py
│   │   ├── health.py
│   │   ├── objects.py
│   │   ├── search.py
│   │   ├── ingest.py
│   │   ├── review.py
│   │   ├── glossary.py
│   │   └── feedback.py
│   ├── services/
│   │   ├── object_service.py
│   │   ├── search_service.py
│   │   ├── retrieval_service.py
│   │   ├── ingestion_service.py
│   │   ├── parser_service.py
│   │   ├── summarizer_service.py
│   │   ├── embedding_service.py
│   │   ├── dependency_service.py
│   │   ├── glossary_service.py
│   │   ├── rule_engine_service.py
│   │   ├── review_service.py
│   │   ├── feedback_service.py
│   │   ├── llm_service.py
│   │   └── mcp_service.py
│   ├── integrations/
│   │   ├── ollama_client.py
│   │   ├── vllm_client.py
│   │   ├── embedding_client.py
│   │   ├── postgres_catalog_client.py
│   │   └── mcp_client.py
│   ├── workers/
│   │   ├── ingest_worker.py
│   │   ├── summary_worker.py
│   │   ├── embedding_worker.py
│   │   └── review_worker.py
│   ├── utils/
│   │   ├── hashing.py
│   │   ├── sql_normalizer.py
│   │   ├── chunking.py
│   │   ├── ranking.py
│   │   ├── time.py
│   │   └── ids.py
│   └── prompts/
│       ├── review/
│       │   ├── technical_review.txt
│       │   ├── business_logic_review.txt
│       │   ├── merged_review.txt
│       │   └── validation_queries.txt
│       ├── retrieval/
│       │   ├── object_summary.txt
│       │   └── glossary_linking.txt
│       └── ingestion/
│           ├── technical_summary.txt
│           └── business_summary.txt
├── tests/
│   ├── conftest.py
│   ├── api/
│   │   ├── test_health.py
│   │   ├── test_objects.py
│   │   ├── test_search.py
│   │   ├── test_ingest.py
│   │   └── test_review.py
│   ├── services/
│   │   ├── test_retrieval_service.py
│   │   ├── test_review_service.py
│   │   ├── test_ingestion_service.py
│   │   └── test_rule_engine_service.py
│   └── repositories/
│       ├── test_object_repo.py
│       └── test_review_repo.py
├── alembic/
│   ├── versions/
│   ├── env.py
│   └── script.py.mako
├── scripts/
│   ├── seed_reference_data.py
│   ├── backfill_embeddings.py
│   ├── run_ingestion.py
│   └── run_review.py
├── docker/
│   ├── api.Dockerfile
│   └── worker.Dockerfile
├── .env.example
├── pyproject.toml
├── alembic.ini
├── docker-compose.yml
└── README.md
```

## Layer responsibilities

Keep routers thin: they should validate input, call a service, and return a response. Business rules should live in named services or use-case functions, while repositories handle DB persistence and query composition. [medium](https://medium.com/@rameshkannanyt0078/structuring-fastapi-with-repository-router-business-logic-models-a-clean-architecture-guide-5ec85443c23c)

Recommended responsibilities:
- `api/`: HTTP-only concerns, request/response, dependency injection.
- `schemas/`: Pydantic request/response contracts, separate from ORM models.
- `services/`: orchestration, review workflow, retrieval logic, business decisions.
- `db/repositories/`: database reads/writes, search SQL, transaction-safe persistence.
- `integrations/`: LLM, embeddings, MCP, and catalog access clients.
- `workers/`: async or batch execution for ingestion and long-running reviews.
- `prompts/`: versioned prompt templates stored as files for auditability.

## Minimal file blueprint

### `app/main.py`

```python
from fastapi import FastAPI
from app.api.v1.api import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.lifespan import lifespan

configure_logging()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)
```

### `app/lifespan.py`

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
```

### `app/api/v1/api.py`

```python
from fastapi import APIRouter
from app.api.v1.endpoints import health, objects, search, ingest, review, glossary, feedback

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(objects.router, prefix="/objects", tags=["objects"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
api_router.include_router(review.router, prefix="/review", tags=["review"])
api_router.include_router(glossary.router, prefix="/glossary", tags=["glossary"])
api_router.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
```

### `app/core/config.py`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "CRM SQL Review Copilot"
    APP_VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"

    DATABASE_URL: str
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GENERATION_MODEL: str = "qwen2.5-coder"
    EMBEDDING_MODEL: str = "nomic-embed-text"

    REVIEW_TOPK_LEXICAL: int = 10
    REVIEW_TOPK_SEMANTIC: int = 10
    REVIEW_TOPK_DEPENDENCY: int = 10

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
```

### `app/db/session.py`

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)

async def get_db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session
```

### `app/api/deps.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.services.review_service import ReviewService
from app.services.retrieval_service import RetrievalService

async def db_session() -> AsyncSession:
    async for session in get_db():
        yield session

def get_review_service(db: AsyncSession) -> ReviewService:
    return ReviewService(db=db)

def get_retrieval_service(db: AsyncSession) -> RetrievalService:
    return RetrievalService(db=db)
```

### `app/schemas/review.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ReviewRequest(BaseModel):
    object_fqn: Optional[str] = None
    sql_text: Optional[str] = None
    review_scope: str = Field(default="manual")
    include_business_logic: bool = True

class ReviewFindingOut(BaseModel):
    severity: str
    category: str
    title: str
    evidence: Optional[str] = None
    impact: Optional[str] = None
    recommendation: Optional[str] = None
    confidence: Optional[float] = None

class ReviewResponse(BaseModel):
    overall_risk: str
    findings: List[ReviewFindingOut]
    validation_queries: List[str] = []
    deployment_checks: List[str] = []
```

### `app/api/v1/endpoints/review.py`

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.review import ReviewRequest, ReviewResponse
from app.db.session import get_db
from app.services.review_service import ReviewService

router = APIRouter()

@router.post("", response_model=ReviewResponse)
async def review_sql(payload: ReviewRequest, db: AsyncSession = Depends(get_db)):
    service = ReviewService(db=db)
    return await service.run_review(payload)
```

### `app/services/review_service.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.review import ReviewRequest, ReviewResponse, ReviewFindingOut
from app.services.retrieval_service import RetrievalService
from app.services.rule_engine_service import RuleEngineService
from app.services.llm_service import LLMService

class ReviewService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.retrieval = RetrievalService(db)
        self.rule_engine = RuleEngineService(db)
        self.llm = LLMService()

    async def run_review(self, payload: ReviewRequest) -> ReviewResponse:
        context = await self.retrieval.build_review_context(payload)
        deterministic_findings = await self.rule_engine.run_checks(payload, context)
        llm_findings = await self.llm.review(payload, context)

        findings = deterministic_findings + llm_findings
        overall_risk = self._derive_overall_risk(findings)

        return ReviewResponse(
            overall_risk=overall_risk,
            findings=[ReviewFindingOut(**f) for f in findings],
            validation_queries=context.get("validation_queries", []),
            deployment_checks=context.get("deployment_checks", []),
        )

    def _derive_overall_risk(self, findings: list[dict]) -> str:
        severities = [f["severity"] for f in findings]
        if "critical" in severities:
            return "critical"
        if "high" in severities:
            return "high"
        if "medium" in severities:
            return "medium"
        return "low"
```

### `app/services/retrieval_service.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.review import ReviewRequest

class RetrievalService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def build_review_context(self, payload: ReviewRequest) -> dict:
        return {
            "object": {},
            "related_objects": [],
            "glossary_terms": [],
            "rules": [],
            "incidents": [],
            "validation_queries": [],
            "deployment_checks": [],
        }
```

### `app/services/llm_service.py`

```python
class LLMService:
    async def review(self, payload, context: dict) -> list[dict]:
        return []
```

### `app/services/rule_engine_service.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession

class RuleEngineService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def run_checks(self, payload, context: dict) -> list[dict]:
        findings = []
        if payload.sql_text and "select *" in payload.sql_text.lower():
            findings.append({
                "severity": "medium",
                "category": "maintainability",
                "title": "Avoid SELECT *",
                "evidence": "Query contains SELECT *",
                "impact": "Schema drift and unnecessary data retrieval risk",
                "recommendation": "Use explicit column names",
                "confidence": 0.99,
            })
        return findings
```

## Suggested repository pattern

Repositories should hide SQLAlchemy query details from services. This matches the common advice that the repository layer encapsulates persistence, while the service layer owns business behavior. [pysquad](https://pysquad.com/blogs/building-maintainable-apis-with-fastapi-best-pract)

Example:

```python
# app/db/repositories/object_repo.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models.db_object import DbObject

class ObjectRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_fqn(self, object_fqn: str) -> DbObject | None:
        result = await self.db.execute(
            select(DbObject).where(DbObject.object_fqn == object_fqn)
        )
        return result.scalar_one_or_none()
```

## ORM model strategy

Keep SQLAlchemy ORM models separate from Pydantic schemas. FastAPI guidance consistently favors this separation because request/response validation and persistence modeling evolve differently over time. [mshaeri](https://mshaeri.com/blog/fastapi-sqlalchemy-best-practices/)

Example ORM base:

```python
# app/db/base.py
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass
```

Example ORM model:

```python
# app/db/models/db_object.py
from sqlalchemy import BigInteger, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.db.base import Base

class DbObject(Base):
    __tablename__ = "db_object"
    __table_args__ = {"schema": "ai_kb"}

    object_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_system_id: Mapped[int] = mapped_column(ForeignKey("ai_kb.source_system.source_system_id"))
    db_name: Mapped[str] = mapped_column(Text, nullable=False)
    schema_name: Mapped[str] = mapped_column(Text, nullable=False)
    object_name: Mapped[str] = mapped_column(Text, nullable=False)
    object_type: Mapped[str] = mapped_column(Text, nullable=False)
    object_fqn: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
```

## Core endpoints to implement first

Start with a small, useful surface area:
- `GET /health`
- `GET /objects/{object_fqn}`
- `GET /objects/{object_fqn}/dependencies`
- `POST /search/context`
- `POST /review`
- `POST /ingest/object`
- `POST /feedback`

This sequencing keeps initial development focused on the core loop: ingest → retrieve → review → feedback.

## Testing layout

Put most tests on services and repositories, not only HTTP endpoints. FastAPI architecture guidance generally recommends testing business logic separately from transport because service-level tests are faster, more precise, and easier to maintain. [pysquad](https://pysquad.com/blogs/building-maintainable-apis-with-fastapi-best-pract)

Suggested tests:
- `services/test_rule_engine_service.py`
- `services/test_retrieval_service.py`
- `services/test_review_service.py`
- `repositories/test_object_repo.py`
- `api/test_review.py`

## Recommended dependencies

Use:
- `fastapi`
- `uvicorn`
- `sqlalchemy`
- `psycopg[binary]`
- `alembic`
- `pydantic`
- `pydantic-settings`
- `httpx`
- `structlog` or standard `logging`
- `pytest`
- `pytest-asyncio`

This set supports a typed FastAPI app with modern SQLAlchemy session handling, environment-based config, and migration support, all of which are consistent with scalable FastAPI project patterns. [github](https://github.com/kumarsonu676/python-fastapi-starter-api-project)

## Practical implementation advice

For your use case, keep these boundaries strict:
- Routers should not build prompts.
- Repositories should not call the LLM.
- Services should not contain raw SQL strings unless they are orchestration-specific.
- Prompt files should live outside code so you can version and compare them easily.

That separation will matter once you start adding multiple review modes, domain-specific prompt variants, and ingestion workers.

The next item in your sequence is **3. ingestion pipeline code**.

---------------------------

Below is a practical **ingestion pipeline code blueprint** for your CRM SQL Review Copilot. The core ideas are: extract metadata from PostgreSQL catalogs and repos, normalize and chunk SQL logically, avoid recomputing embeddings when content hashes have not changed, and run ingestion asynchronously so the API stays responsive. [nstarxinc](https://nstarxinc.com/blog/from-data-lake-to-rag-factory-the-technical-view-building-incremental-embedding-pipelines-without-melting-your-cloud-bill/)

## Pipeline shape

The ingestion flow should be: **discover → extract → normalize → hash → summarize → chunk → embed → upsert → link dependencies**. Avoid re-embedding unchanged content by using content hashes, because incremental embedding pipelines save cost and reduce unnecessary work. [medium](https://medium.com/@shekhar.manna83/rag-architecture-best-practice-vector-database-ingestion-6a7aecaa5ae4)

A good first implementation uses Python services plus worker entry points. PostgreSQL metadata should come from `information_schema`, `pg_catalog`, and helper functions like `pg_get_functiondef`, `pg_get_viewdef`, and `pg_indexes`, because those are the standard ways to retrieve object definitions and catalog metadata. [dev](https://dev.to/itsjjpowell/learn-more-about-your-database-with-postgres-information-schema-2pid)

## Suggested files

```text
app/
  services/
    ingestion_service.py
    parser_service.py
    summarizer_service.py
    embedding_service.py
    dependency_service.py
  integrations/
    postgres_catalog_client.py
    embedding_client.py
    ollama_client.py
  utils/
    hashing.py
    sql_normalizer.py
    chunking.py
  workers/
    ingest_worker.py
scripts/
  run_ingestion.py
```

## Core models

Use simple dataclasses internally for extraction and transformation.

### `app/services/contracts.py`

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ExtractedObject:
    source_system_id: int
    db_name: str
    schema_name: str
    object_name: str
    object_type: str
    object_fqn: str
    object_signature: str | None
    source_sql: str
    normalized_sql: str
    source_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ObjectDocument:
    doc_kind: str
    title: str | None
    content: str
    content_hash: str
    chunk_no: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

## Hashing and normalization

Use stable normalization before hashing so irrelevant formatting changes do not trigger re-ingestion.

### `app/utils/hashing.py`

```python
import hashlib

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

### `app/utils/sql_normalizer.py`

```python
import re

def normalize_sql(sql: str) -> str:
    text = sql.strip()
    text = re.sub(r'--.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()
```

This is intentionally simple for V1. Later you can replace it with SQLGlot or a PostgreSQL-aware parser for stronger canonicalization.

## Logical chunking

Chunk by meaning, not fixed token count, because SQL review quality improves when chunks align with logical blocks rather than arbitrary windows.

### `app/utils/chunking.py`

```python
import re
from app.services.contracts import ObjectDocument
from app.utils.hashing import sha256_text

CTE_SPLIT = re.compile(r'\bwith\b', re.IGNORECASE)
BLOCK_SPLIT = re.compile(
    r'(\bselect\b|\binsert\b|\bupdate\b|\bdelete\b|\bcreate\b|\bexception\b|\bbegin\b|\bend\b)',
    re.IGNORECASE
)

def chunk_sql_logically(sql: str) -> list[ObjectDocument]:
    chunks: list[ObjectDocument] = []
    parts = [p.strip() for p in re.split(BLOCK_SPLIT, sql) if p and p.strip()]
    merged: list[str] = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts):
            merged.append(parts[i] + " " + parts[i + 1])
            i += 2
        else:
            merged.append(parts[i])
            i += 1

    for idx, part in enumerate(merged, start=1):
        chunks.append(
            ObjectDocument(
                doc_kind="logic_chunk",
                title=f"SQL logic chunk {idx}",
                content=part.strip(),
                content_hash=sha256_text(part.strip()),
                chunk_no=idx,
                metadata={"chunk_strategy": "logical_block"},
            )
        )
    return chunks
```

## PostgreSQL catalog extraction

Use `information_schema` for portable metadata where possible, and `pg_catalog` / `pg_get_*` functions for PostgreSQL-specific definitions. [stackoverflow](https://stackoverflow.com/questions/12148914/get-definition-of-function-sequence-type-etc-in-postgresql-with-sql-query)

### `app/integrations/postgres_catalog_client.py`

```python
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

class PostgresCatalogClient:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def fetch_views(self, schemas: list[str]) -> list[dict]:
        stmt = text("""
            SELECT schemaname, viewname, definition
            FROM pg_views
            WHERE schemaname = ANY(:schemas)
        """)
        result = await self.db.execute(stmt, {"schemas": schemas})
        return [dict(row._mapping) for row in result]

    async def fetch_materialized_views(self, schemas: list[str]) -> list[dict]:
        stmt = text("""
            SELECT schemaname, matviewname, definition
            FROM pg_matviews
            WHERE schemaname = ANY(:schemas)
        """)
        result = await self.db.execute(stmt, {"schemas": schemas})
        return [dict(row._mapping) for row in result]

    async def fetch_functions(self, schemas: list[str]) -> list[dict]:
        stmt = text("""
            SELECT
                n.nspname AS schema_name,
                p.proname AS object_name,
                pg_get_functiondef(p.oid) AS definition,
                pg_get_function_identity_arguments(p.oid) AS identity_args
            FROM pg_proc p
            JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = ANY(:schemas)
        """)
        result = await self.db.execute(stmt, {"schemas": schemas})
        return [dict(row._mapping) for row in result]

    async def fetch_tables(self, schemas: list[str]) -> list[dict]:
        stmt = text("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema = ANY(:schemas)
              AND table_type = 'BASE TABLE'
        """)
        result = await self.db.execute(stmt, {"schemas": schemas})
        return [dict(row._mapping) for row in result]

    async def fetch_columns(self, schemas: list[str]) -> list[dict]:
        stmt = text("""
            SELECT table_schema, table_name, column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = ANY(:schemas)
            ORDER BY table_schema, table_name, ordinal_position
        """)
        result = await self.db.execute(stmt, {"schemas": schemas})
        return [dict(row._mapping) for row in result]

    async def fetch_indexes(self, schemas: list[str]) -> list[dict]:
        stmt = text("""
            SELECT schemaname, tablename, indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = ANY(:schemas)
        """)
        result = await self.db.execute(stmt, {"schemas": schemas})
        return [dict(row._mapping) for row in result]
```

## Parser service

Create `ExtractedObject` instances from raw catalog data.

### `app/services/parser_service.py`

```python
from collections import defaultdict
from app.services.contracts import ExtractedObject
from app.utils.hashing import sha256_text
from app.utils.sql_normalizer import normalize_sql

class ParserService:
    def build_table_objects(
        self,
        source_system_id: int,
        db_name: str,
        tables: list[dict],
        columns: list[dict],
    ) -> list[ExtractedObject]:
        grouped = defaultdict(list)
        for col in columns:
            grouped[(col["table_schema"], col["table_name"])].append(col)

        objects: list[ExtractedObject] = []
        for row in tables:
            schema_name = row["table_schema"]
            object_name = row["table_name"]
            cols = grouped[(schema_name, object_name)]

            ddl_lines = [f'CREATE TABLE "{schema_name}"."{object_name}" (']
            for col in cols:
                line = f'  "{col["column_name"]}" {col["data_type"]}'
                if col["is_nullable"] == "NO":
                    line += " NOT NULL"
                if col["column_default"]:
                    line += f" DEFAULT {col['column_default']}"
                ddl_lines.append(line + ",")
            if len(ddl_lines) > 1:
                ddl_lines[-1] = ddl_lines[-1].rstrip(",")
            ddl_lines.append(");")
            ddl = "\n".join(ddl_lines)

            normalized = normalize_sql(ddl)
            objects.append(
                ExtractedObject(
                    source_system_id=source_system_id,
                    db_name=db_name,
                    schema_name=schema_name,
                    object_name=object_name,
                    object_type="table",
                    object_fqn=f"{schema_name}.{object_name}",
                    object_signature=None,
                    source_sql=ddl,
                    normalized_sql=normalized,
                    source_hash=sha256_text(normalized),
                    metadata={"column_count": len(cols)},
                )
            )
        return objects

    def build_view_objects(self, source_system_id: int, db_name: str, rows: list[dict]) -> list[ExtractedObject]:
        objects = []
        for row in rows:
            schema_name = row["schemaname"]
            object_name = row["viewname"]
            ddl = f'CREATE VIEW "{schema_name}"."{object_name}" AS\n{row["definition"]}'
            normalized = normalize_sql(ddl)
            objects.append(
                ExtractedObject(
                    source_system_id=source_system_id,
                    db_name=db_name,
                    schema_name=schema_name,
                    object_name=object_name,
                    object_type="view",
                    object_fqn=f"{schema_name}.{object_name}",
                    object_signature=None,
                    source_sql=ddl,
                    normalized_sql=normalized,
                    source_hash=sha256_text(normalized),
                    metadata={},
                )
            )
        return objects

    def build_function_objects(self, source_system_id: int, db_name: str, rows: list[dict]) -> list[ExtractedObject]:
        objects = []
        for row in rows:
            schema_name = row["schema_name"]
            object_name = row["object_name"]
            signature = row.get("identity_args") or ""
            ddl = row["definition"]
            normalized = normalize_sql(ddl)
            objects.append(
                ExtractedObject(
                    source_system_id=source_system_id,
                    db_name=db_name,
                    schema_name=schema_name,
                    object_name=object_name,
                    object_type="function",
                    object_fqn=f"{schema_name}.{object_name}",
                    object_signature=signature,
                    source_sql=ddl,
                    normalized_sql=normalized,
                    source_hash=sha256_text(normalized),
                    metadata={"identity_args": signature},
                )
            )
        return objects
```

## Summarizer service

Use your local LLM to create short technical and business summaries. Keep them deterministic and bounded.

### `app/services/summarizer_service.py`

```python
from app.services.contracts import ExtractedObject, ObjectDocument
from app.utils.hashing import sha256_text

class SummarizerService:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def build_documents(self, obj: ExtractedObject) -> list[ObjectDocument]:
        technical_summary = await self.llm_client.generate(
            system_prompt="Summarize SQL objects for technical retrieval.",
            user_prompt=(
                f"Object type: {obj.object_type}\n"
                f"Object FQN: {obj.object_fqn}\n"
                f"SQL:\n{obj.source_sql}\n\n"
                "Return a concise 5-8 sentence technical summary covering purpose, joins, writes, filters, aggregation, risk areas."
            ),
            temperature=0.1,
        )

        business_summary = await self.llm_client.generate(
            system_prompt="Summarize SQL objects for business-domain retrieval.",
            user_prompt=(
                f"Object type: {obj.object_type}\n"
                f"Object FQN: {obj.object_fqn}\n"
                f"SQL:\n{obj.source_sql}\n\n"
                "Infer probable business purpose in CRM terms. Mention likely grain, entities, and business processes. If uncertain, state uncertainty."
            ),
            temperature=0.1,
        )

        return [
            ObjectDocument(
                doc_kind="raw_sql",
                title=f"{obj.object_fqn} raw sql",
                content=obj.source_sql,
                content_hash=sha256_text(obj.source_sql),
            ),
            ObjectDocument(
                doc_kind="normalized_sql",
                title=f"{obj.object_fqn} normalized sql",
                content=obj.normalized_sql,
                content_hash=sha256_text(obj.normalized_sql),
            ),
            ObjectDocument(
                doc_kind="technical_summary",
                title=f"{obj.object_fqn} technical summary",
                content=technical_summary,
                content_hash=sha256_text(technical_summary),
            ),
            ObjectDocument(
                doc_kind="business_summary",
                title=f"{obj.object_fqn} business summary",
                content=business_summary,
                content_hash=sha256_text(business_summary),
            ),
        ]
```

## Embedding client and service

Only embed docs whose `content_hash` is new or changed. This is one of the biggest practical wins in ingestion systems. [index-management](https://www.index-management.org/embedding-ingestion-pipeline-engineering/)

### `app/integrations/embedding_client.py`

```python
class EmbeddingClient:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
```

### `app/services/embedding_service.py`

```python
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.contracts import ObjectDocument

class EmbeddingService:
    def __init__(self, db: AsyncSession, embedding_client, embedding_model_id: int):
        self.db = db
        self.embedding_client = embedding_client
        self.embedding_model_id = embedding_model_id

    async def filter_docs_needing_embeddings(self, docs: list[tuple[int, ObjectDocument]]) -> list[tuple[int, ObjectDocument]]:
        filtered: list[tuple[int, ObjectDocument]] = []
        for doc_id, doc in docs:
            stmt = text("""
                SELECT 1
                FROM ai_kb.object_doc_embedding
                WHERE doc_id = :doc_id
                  AND embedding_model_id = :embedding_model_id
                  AND content_hash = :content_hash
                LIMIT 1
            """)
            result = await self.db.execute(stmt, {
                "doc_id": doc_id,
                "embedding_model_id": self.embedding_model_id,
                "content_hash": doc.content_hash,
            })
            exists = result.scalar_one_or_none()
            if not exists:
                filtered.append((doc_id, doc))
        return filtered

    async def embed_and_store(self, docs: list[tuple[int, ObjectDocument]]) -> int:
        if not docs:
            return 0

        texts = [doc.content for _, doc in docs]
        vectors = await self.embedding_client.embed(texts)

        for (doc_id, doc), vector in zip(docs, vectors):
            stmt = text("""
                INSERT INTO ai_kb.object_doc_embedding (doc_id, embedding_model_id, content_hash, embedding)
                VALUES (:doc_id, :embedding_model_id, :content_hash, :embedding)
                ON CONFLICT (doc_id, embedding_model_id, content_hash) DO NOTHING
            """)
            await self.db.execute(stmt, {
                "doc_id": doc_id,
                "embedding_model_id": self.embedding_model_id,
                "content_hash": doc.content_hash,
                "embedding": vector,
            })

        return len(docs)
```

## Dependency extraction

Start with a heuristic extractor, then improve later with a parser. Even a simple regex-based pass gives useful first-hop lineage.

### `app/services/dependency_service.py`

```python
import re

FROM_JOIN_PATTERN = re.compile(r'\b(from|join|update|into)\s+("?[\w]+"?\.)?"?([\w]+)"?', re.IGNORECASE)

class DependencyService:
    def extract_dependencies(self, schema_name: str, sql: str) -> list[dict]:
        matches = FROM_JOIN_PATTERN.findall(sql)
        deps = []
        seen = set()

        for keyword, schema_part, object_name in matches:
            dep_schema = schema_part.replace('"', '').replace('.', '') if schema_part else schema_name
            fqn = f"{dep_schema}.{object_name}"
            dep_type = "reads"
            if keyword.lower() in {"into", "update"}:
                dep_type = "writes"

            key = (fqn, dep_type)
            if key not in seen:
                seen.add(key)
                deps.append({
                    "to_object_fqn": fqn,
                    "dependency_type": dep_type,
                    "confidence": 0.6,
                    "discovered_by": "parser",
                })
        return deps
```

## Repository helpers for upsert

Use repository methods for clean DB writes.

### `app/db/repositories/object_repo.py`

```python
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.contracts import ExtractedObject, ObjectDocument

class ObjectRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def upsert_object(self, obj: ExtractedObject) -> int:
        stmt = text("""
            INSERT INTO ai_kb.db_object (
                source_system_id, db_name, schema_name, object_name, object_type, object_signature, object_fqn, last_seen_at
            )
            VALUES (
                :source_system_id, :db_name, :schema_name, :object_name, :object_type, :object_signature, :object_fqn, now()
            )
            ON CONFLICT (source_system_id, object_fqn)
            DO UPDATE SET
                db_name = EXCLUDED.db_name,
                schema_name = EXCLUDED.schema_name,
                object_name = EXCLUDED.object_name,
                object_type = EXCLUDED.object_type,
                object_signature = EXCLUDED.object_signature,
                last_seen_at = now(),
                updated_at = now()
            RETURNING object_id
        """)
        result = await self.db.execute(stmt, obj.__dict__)
        return result.scalar_one()

    async def get_current_hash(self, object_id: int) -> str | None:
        stmt = text("""
            SELECT source_hash
            FROM ai_kb.db_object_version
            WHERE object_id = :object_id AND is_current = TRUE
        """)
        result = await self.db.execute(stmt, {"object_id": object_id})
        return result.scalar_one_or_none()

    async def insert_new_version_if_changed(self, object_id: int, obj: ExtractedObject) -> int | None:
        current_hash = await self.get_current_hash(object_id)
        if current_hash == obj.source_hash:
            return None

        await self.db.execute(text("""
            UPDATE ai_kb.db_object_version
            SET is_current = FALSE, valid_to = now()
            WHERE object_id = :object_id AND is_current = TRUE
        """), {"object_id": object_id})

        version_no_result = await self.db.execute(text("""
            SELECT COALESCE(MAX(version_no), 0) + 1
            FROM ai_kb.db_object_version
            WHERE object_id = :object_id
        """), {"object_id": object_id})
        version_no = version_no_result.scalar_one()

        stmt = text("""
            INSERT INTO ai_kb.db_object_version (
                object_id, version_no, source_hash, source_sql, normalized_sql, is_current
            )
            VALUES (
                :object_id, :version_no, :source_hash, :source_sql, :normalized_sql, TRUE
            )
            RETURNING object_version_id
        """)
        result = await self.db.execute(stmt, {
            "object_id": object_id,
            "version_no": version_no,
            "source_hash": obj.source_hash,
            "source_sql": obj.source_sql,
            "normalized_sql": obj.normalized_sql,
        })
        return result.scalar_one()

    async def insert_docs(self, object_id: int, object_version_id: int | None, docs: list[ObjectDocument]) -> list[tuple[int, ObjectDocument]]:
        inserted = []
        for doc in docs:
            stmt = text("""
                INSERT INTO ai_kb.object_doc (
                    object_id, object_version_id, doc_kind, chunk_no, title, content, content_hash, metadata_json
                )
                VALUES (
                    :object_id, :object_version_id, :doc_kind, :chunk_no, :title, :content, :content_hash, :metadata_json
                )
                ON CONFLICT (object_id, doc_kind, content_hash, chunk_no)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    metadata_json = EXCLUDED.metadata_json
                RETURNING doc_id
            """)
            result = await self.db.execute(stmt, {
                "object_id": object_id,
                "object_version_id": object_version_id,
                "doc_kind": doc.doc_kind,
                "chunk_no": doc.chunk_no,
                "title": doc.title,
                "content": doc.content,
                "content_hash": doc.content_hash,
                "metadata_json": doc.metadata,
            })
            doc_id = result.scalar_one()
            inserted.append((doc_id, doc))
        return inserted

    async def replace_dependencies(self, object_id: int, deps: list[dict]) -> None:
        await self.db.execute(text("""
            DELETE FROM ai_kb.object_dependency
            WHERE from_object_id = :object_id AND discovered_by = 'parser'
        """), {"object_id": object_id})

        for dep in deps:
            lookup = await self.db.execute(text("""
                SELECT object_id
                FROM ai_kb.db_object
                WHERE object_fqn = :object_fqn
                LIMIT 1
            """), {"object_fqn": dep["to_object_fqn"]})
            to_object_id = lookup.scalar_one_or_none()
            if not to_object_id:
                continue

            await self.db.execute(text("""
                INSERT INTO ai_kb.object_dependency (
                    from_object_id, to_object_id, dependency_type, confidence, discovered_by
                )
                VALUES (
                    :from_object_id, :to_object_id, :dependency_type, :confidence, :discovered_by
                )
                ON CONFLICT (from_object_id, to_object_id, dependency_type, discovered_by) DO NOTHING
            """), {
                "from_object_id": object_id,
                "to_object_id": to_object_id,
                "dependency_type": dep["dependency_type"],
                "confidence": dep["confidence"],
                "discovered_by": dep["discovered_by"],
            })
```

## Ingestion service

This is the orchestration core.

### `app/services/ingestion_service.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession
from app.integrations.postgres_catalog_client import PostgresCatalogClient
from app.db.repositories.object_repo import ObjectRepository
from app.services.parser_service import ParserService
from app.services.summarizer_service import SummarizerService
from app.services.embedding_service import EmbeddingService
from app.services.dependency_service import DependencyService
from app.utils.chunking import chunk_sql_logically

class IngestionService:
    def __init__(
        self,
        db: AsyncSession,
        catalog_client: PostgresCatalogClient,
        parser_service: ParserService,
        summarizer_service: SummarizerService,
        embedding_service: EmbeddingService,
        dependency_service: DependencyService,
    ):
        self.db = db
        self.catalog_client = catalog_client
        self.parser_service = parser_service
        self.summarizer_service = summarizer_service
        self.embedding_service = embedding_service
        self.dependency_service = dependency_service
        self.object_repo = ObjectRepository(db)

    async def ingest_database(self, source_system_id: int, db_name: str, schemas: list[str]) -> dict:
        tables = await self.catalog_client.fetch_tables(schemas)
        columns = await self.catalog_client.fetch_columns(schemas)
        views = await self.catalog_client.fetch_views(schemas)
        functions = await self.catalog_client.fetch_functions(schemas)

        objects = []
        objects.extend(self.parser_service.build_table_objects(source_system_id, db_name, tables, columns))
        objects.extend(self.parser_service.build_view_objects(source_system_id, db_name, views))
        objects.extend(self.parser_service.build_function_objects(source_system_id, db_name, functions))

        scanned = len(objects)
        changed = 0
        embedded = 0

        for obj in objects:
            object_id = await self.object_repo.upsert_object(obj)
            object_version_id = await self.object_repo.insert_new_version_if_changed(object_id, obj)

            if object_version_id is None:
                continue

            changed += 1

            summary_docs = await self.summarizer_service.build_documents(obj)
            chunk_docs = chunk_sql_logically(obj.source_sql)
            docs = summary_docs + chunk_docs

            inserted_docs = await self.object_repo.insert_docs(object_id, object_version_id, docs)
            docs_to_embed = await self.embedding_service.filter_docs_needing_embeddings(inserted_docs)
            embedded += await self.embedding_service.embed_and_store(docs_to_embed)

            deps = self.dependency_service.extract_dependencies(obj.schema_name, obj.source_sql)
            await self.object_repo.replace_dependencies(object_id, deps)

        await self.db.commit()
        return {
            "objects_scanned": scanned,
            "objects_changed": changed,
            "docs_embedded": embedded,
        }
```

## Worker entry point

Run ingestion outside the request path for scale and responsiveness. Background task workers are a better fit than long-running API requests for heavier ingestion jobs. [medium](https://medium.com/@hadiyolworld007/fastapi-and-celery-at-scale-building-async-task-pipelines-that-fly-ef185527abf2)

### `app/workers/ingest_worker.py`

```python
import asyncio
from app.db.session import SessionLocal
from app.integrations.postgres_catalog_client import PostgresCatalogClient
from app.integrations.ollama_client import OllamaClient
from app.integrations.embedding_client_impl import LocalEmbeddingClient
from app.services.parser_service import ParserService
from app.services.summarizer_service import SummarizerService
from app.services.embedding_service import EmbeddingService
from app.services.dependency_service import DependencyService
from app.services.ingestion_service import IngestionService

async def run_ingestion(source_system_id: int, db_name: str, schemas: list[str], embedding_model_id: int):
    async with SessionLocal() as db:
        catalog = PostgresCatalogClient(db)
        llm_client = OllamaClient()
        embedding_client = LocalEmbeddingClient()

        service = IngestionService(
            db=db,
            catalog_client=catalog,
            parser_service=ParserService(),
            summarizer_service=SummarizerService(llm_client),
            embedding_service=EmbeddingService(db, embedding_client, embedding_model_id),
            dependency_service=DependencyService(),
        )
        result = await service.ingest_database(source_system_id, db_name, schemas)
        return result

if __name__ == "__main__":
    asyncio.run(run_ingestion(1, "crm", ["public", "sales", "analytics"], 1))
```

## Example Ollama integration

### `app/integrations/ollama_client.py`

```python
import httpx
from app.core.config import settings

class OllamaClient:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.GENERATION_MODEL

    async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "prompt": f"System:\n{system_prompt}\n\nUser:\n{user_prompt}",
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
```

## Example local embedding integration

```python
# app/integrations/embedding_client_impl.py
import httpx
from app.core.config import settings
from app.integrations.embedding_client import EmbeddingClient

class LocalEmbeddingClient(EmbeddingClient):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        async with httpx.AsyncClient(timeout=120) as client:
            for text in texts:
                payload = {"model": settings.EMBEDDING_MODEL, "prompt": text}
                resp = await client.post(f"{settings.OLLAMA_BASE_URL}/api/embeddings", json=payload)
                resp.raise_for_status()
                vectors.append(resp.json()["embedding"])
        return vectors
```

## Script entry point

### `scripts/run_ingestion.py`

```python
import asyncio
from app.workers.ingest_worker import run_ingestion

if __name__ == "__main__":
    result = asyncio.run(
        run_ingestion(
            source_system_id=1,
            db_name="crm",
            schemas=["public", "sales", "analytics"],
            embedding_model_id=1,
        )
    )
    print(result)
```

## What to ingest first

For the first iteration, ingest only:
- tables,
- views,
- functions/procedures,
- indexes,
- glossary entries.

Then add:
- job definitions,
- migration history,
- API-to-object usage mapping,
- incident summaries,
- past review comments.

This staged approach keeps the pipeline simpler while still making retrieval useful.

## Recommended improvements after V1

After the first working version, improve these areas:

- Replace regex dependency extraction with SQLGlot or a PostgreSQL-aware parser.
- Add ingestion runs and object-level status tracking in `ai_kb.ingestion_run`.
- Batch embeddings for throughput if your embedding endpoint supports it.
- Add retry and dead-letter handling for failed summary/embedding steps.
- Add source-specific ingestors:
  - `RepoSqlIngestor`
  - `PostgresCatalogIngestor`
  - `AirflowSqlIngestor`
  - `FastApiUsageIngestor`

## Operational advice

A few engineering rules matter here:
- Never re-embed unchanged docs; use `content_hash` aggressively. [nstarxinc](https://nstarxinc.com/blog/from-data-lake-to-rag-factory-the-technical-view-building-incremental-embedding-pipelines-without-melting-your-cloud-bill/)
- Keep ingestion idempotent with upserts and versioning.
- Commit in batches if your object count is large.
- Do not run extraction queries against restricted production paths without read-only users and allowlisted schemas; PostgreSQL metadata should be read through catalog views and helper functions, not by mutating system catalogs. [runebook](https://runebook.dev/en/docs/postgresql/ddl-schemas/id-1.5.4.12.11.2)
- Keep API-triggered ingestion lightweight; hand off large syncs to workers, which matches common FastAPI background processing patterns for heavier jobs. [medium](https://medium.com/@bhagyarana80/fastapi-background-tasks-at-scale-handling-1m-asynchronous-side-jobs-d5920f14a473)

The next item in your sequence is **4. the complete review prompt pack**.

----------------
Below is the **complete review prompt pack** for your CRM SQL Review Copilot. The pack is designed around structured prompts, bounded scope, explicit evidence requirements, and machine-readable output, which are the patterns that consistently improve code-review quality and reduce noisy feedback. [gresearch](https://www.gresearch.com/news/building-a-code-review-tool-the-llm-patterns-that-actually-work/)

## How to use this pack

Use these prompts as separate passes rather than one giant prompt. Multi-pass review improves consistency because extraction, judgment, and final synthesis are easier to control when split into focused steps, and structured output makes the results easier to store, filter, and compare. [github](https://github.com/orgs/community/discussions/186473)

Recommended sequence:
1. Object summary prompt
2. Retrieval query-building prompt
3. Technical SQL review prompt
4. Business-logic review prompt
5. Merge-and-deduplicate prompt
6. Validation-query prompt
7. Feedback-learning prompt

***

## 1. Object summary prompt

**Purpose:** generate compact retrieval-friendly summaries during ingestion.

```text
ROLE
You are a senior PostgreSQL engineer documenting SQL objects for an internal retrieval system.

TASK
Summarize the SQL object below so it can be retrieved later for code review, dependency analysis, and business-logic validation.

INSTRUCTIONS
- Identify the object’s likely purpose.
- Describe what data it reads, writes, joins, aggregates, or filters.
- Mention likely grain (row-level, invoice-level, order-level, distributor-month, etc.) if inferable.
- Mention likely downstream consumers (API, ETL, dashboard, report) only if strongly supported by the SQL.
- Call out risky areas: temp tables, window functions, joins, dedupe logic, upserts, exception handling, status filters.
- If something is uncertain, explicitly say “uncertain”.

OUTPUT RULES
- Return valid JSON only.
- Keep each field concise.
- Do not invent business meaning that is not supported by the SQL.

OUTPUT SCHEMA
{
  "object_purpose": "string",
  "reads_from": ["string"],
  "writes_to": ["string"],
  "joins": ["string"],
  "filters": ["string"],
  "aggregations": ["string"],
  "grain": "string",
  "risk_areas": ["string"],
  "uncertainties": ["string"]
}

INPUT
Object type: {{object_type}}
Object FQN: {{object_fqn}}
SQL:
{{sql_text}}
```

***

## 2. Retrieval query-building prompt

**Purpose:** derive focused retrieval questions from the object under review.

```text
ROLE
You are a retrieval planner for a SQL review system.

TASK
Given the SQL object and review goal, produce the smallest useful set of retrieval queries needed to gather context for review.

INSTRUCTIONS
- Produce queries for:
  1. same object history,
  2. related objects/dependencies,
  3. business glossary terms,
  4. similar incidents,
  5. similar prior review findings.
- Queries must be concise and retrieval-oriented.
- Prefer exact identifiers when available.
- Include conceptual queries only when exact names are insufficient.
- Do not produce more than 12 total queries.

OUTPUT RULES
- Return valid JSON only.

OUTPUT SCHEMA
{
  "history_queries": ["string"],
  "dependency_queries": ["string"],
  "glossary_queries": ["string"],
  "incident_queries": ["string"],
  "review_queries": ["string"]
}

INPUT
Review goal: {{review_goal}}
Object FQN: {{object_fqn}}
Object type: {{object_type}}
SQL:
{{sql_text}}
```

***

## 3. Technical SQL review prompt

**Purpose:** perform the deterministic + LLM technical pass.

```text
ROLE
You are a senior PostgreSQL code reviewer.

TASK
Review the SQL object for technical quality only.

REVIEW FOCUS
Check for:
- correctness bugs,
- NULL handling issues,
- transaction and exception handling gaps,
- performance risks,
- join/cardinality problems,
- unnecessary scans or broad filters,
- anti-patterns,
- indexing implications,
- maintainability issues that materially affect correctness or performance.

DO NOT
- Do not comment on style-only nits unless they affect clarity or correctness.
- Do not invent missing schema details.
- Do not assume production row counts unless provided in context.
- Do not report an issue unless there is concrete evidence in the SQL or retrieved context.

EVIDENCE RULE
Every finding must include:
- exact line range or logical block reference,
- evidence snippet or pattern,
- why it is risky,
- a concrete fix.

SEVERITY RUBRIC
- critical: likely data corruption, incorrect results, dangerous DML, or production outage risk
- high: strong correctness or major performance risk
- medium: meaningful maintainability or moderate performance risk
- low: minor improvement or preventive guidance
- info: observation only

OUTPUT RULES
- Return valid JSON only.
- Maximum 10 findings.
- If evidence is insufficient, return an empty findings array.

OUTPUT SCHEMA
{
  "overall_risk": "critical|high|medium|low|info",
  "findings": [
    {
      "category": "correctness|performance|maintainability|security|operability",
      "severity": "critical|high|medium|low|info",
      "title": "string",
      "line_start": 0,
      "line_end": 0,
      "evidence": "string",
      "why_it_matters": "string",
      "recommendation": "string",
      "example_fix_sql": "string",
      "confidence": 0.0
    }
  ]
}

INPUT
OBJECT_UNDER_REVIEW
{{object_sql}}

OBJECT_METADATA
{{object_metadata}}

RELATED_OBJECTS
{{related_objects}}

RUNTIME_CONTEXT
{{runtime_context}}

CHECKLIST_RULES
{{technical_rules}}
```

***

## 4. Business-logic review prompt

**Purpose:** review logic against CRM semantics, grain, and business rules.

```text
ROLE
You are a senior data platform architect reviewing SQL for CRM business-rule correctness.

TASK
Review the SQL object for business-logic correctness using the glossary, related objects, and incidents provided.

REVIEW FOCUS
Check whether:
- the object’s grain matches the intended business grain,
- joins can cause duplicate counting,
- returns/credits/reversals are handled in the correct stage,
- status filters align with business definitions,
- date logic matches reporting/settlement periods,
- the object duplicates logic already implemented elsewhere,
- naming and output columns match real business meaning.

IMPORTANT
- Distinguish verified facts from inference.
- If a claim depends on business context, cite the specific glossary or related object evidence.
- If business evidence is weak, mark the finding as uncertain instead of overstating it.

OUTPUT RULES
- Return valid JSON only.
- Maximum 8 findings.
- No finding without evidence from glossary, related objects, or incident history.

OUTPUT SCHEMA
{
  "overall_business_risk": "critical|high|medium|low|info",
  "findings": [
    {
      "category": "business_logic",
      "severity": "critical|high|medium|low|info",
      "title": "string",
      "evidence_source": "glossary|related_object|incident|combined",
      "evidence": "string",
      "business_risk": "string",
      "recommendation": "string",
      "validation_needed": "string",
      "confidence": 0.0
    }
  ]
}

INPUT
OBJECT_UNDER_REVIEW
{{object_sql}}

BUSINESS_GLOSSARY
{{business_glossary}}

RELATED_OBJECTS
{{related_objects}}

INCIDENT_HISTORY
{{incident_history}}

BUSINESS_RULES
{{business_rules}}
```

***

## 5. PR diff review prompt

**Purpose:** review a change, not just a standalone object.

```text
ROLE
You are a senior PostgreSQL reviewer analyzing a code diff.

TASK
Review the diff for regressions and unintended semantic changes.

INSTRUCTIONS
- Compare old and new behavior.
- Focus on changed logic, not unchanged code.
- Identify changed joins, filters, grouping, DML semantics, exception handling, and performance characteristics.
- Flag backwards-incompatible output changes.
- Do not restate the diff; identify risk.

EVIDENCE RULE
Every finding must reference:
- file/object,
- line range from the diff,
- specific changed pattern,
- probable effect.

OUTPUT RULES
- Return valid JSON only.
- Maximum 10 findings.

OUTPUT SCHEMA
{
  "overall_risk": "critical|high|medium|low|info",
  "findings": [
    {
      "severity": "critical|high|medium|low|info",
      "category": "correctness|performance|business_logic|security|maintainability",
      "location": "string",
      "changed_pattern": "string",
      "risk": "string",
      "recommendation": "string",
      "confidence": 0.0
    }
  ]
}

INPUT
DIFF
{{sql_diff}}

OLD_CONTEXT
{{old_object_context}}

NEW_CONTEXT
{{new_object_context}}

RELEVANT_RULES
{{rules}}
```

***

## 6. Merge and deduplicate prompt

**Purpose:** combine deterministic findings, technical review findings, and business-logic findings into one final set. Structured post-processing is important because structured output alone does not guarantee correctness, so a merger/critic step helps remove duplicates and ungrounded claims. [gresearch](https://www.gresearch.com/news/building-a-code-review-tool-the-llm-patterns-that-actually-work/)

```text
ROLE
You are a review synthesizer.

TASK
Merge the three finding lists below into one final review.

INSTRUCTIONS
- Deduplicate overlapping findings.
- Prefer the finding with stronger evidence.
- If two findings describe the same issue, keep one merged version.
- Drop speculative findings without evidence.
- Preserve severity only if justified by the supplied rubric.
- Keep total findings to a maximum of 10.

OUTPUT RULES
- Return valid JSON only.

OUTPUT SCHEMA
{
  "overall_risk": "critical|high|medium|low|info",
  "findings": [
    {
      "severity": "critical|high|medium|low|info",
      "category": "correctness|performance|business_logic|security|maintainability|operability",
      "title": "string",
      "evidence": "string",
      "impact": "string",
      "recommendation": "string",
      "confidence": 0.0,
      "sources": ["deterministic|technical|business"]
    }
  ]
}

INPUT
SEVERITY_RUBRIC
{{severity_rubric}}

DETERMINISTIC_FINDINGS
{{deterministic_findings}}

TECHNICAL_FINDINGS
{{technical_findings}}

BUSINESS_FINDINGS
{{business_findings}}
```

***

## 7. Validation-query prompt

**Purpose:** produce concrete SQL checks to verify risky logic in staging or pre-prod.

```text
ROLE
You are a PostgreSQL validation engineer.

TASK
Generate SQL validation queries and rollout checks for the findings below.

INSTRUCTIONS
- Produce only safe read-only SQL.
- Focus on validating correctness and business-rule assumptions.
- Prefer aggregate comparisons, duplicate detection, row-count checks, NULL audits, and before/after comparisons.
- If a validation query requires unavailable tables, say so explicitly.
- Do not generate destructive SQL.

OUTPUT RULES
- Return valid JSON only.
- Maximum 8 validation queries and 8 deployment checks.

OUTPUT SCHEMA
{
  "validation_queries": [
    {
      "title": "string",
      "purpose": "string",
      "sql": "string"
    }
  ],
  "deployment_checks": [
    "string"
  ]
}

INPUT
OBJECT_UNDER_REVIEW
{{object_sql}}

FINAL_FINDINGS
{{final_findings}}

RELATED_OBJECTS
{{related_objects}}
```

***

## 8. Feedback-learning prompt

**Purpose:** convert human reviewer feedback into prompt/rule improvements.

```text
ROLE
You are a prompt and rule improvement analyst.

TASK
Analyze review feedback and suggest improvements to rules, retrieval, or prompting.

INSTRUCTIONS
- Identify whether the miss was caused by:
  - missing retrieval context,
  - weak business glossary coverage,
  - poor severity calibration,
  - duplicate findings,
  - hallucinated reasoning,
  - missing deterministic rule.
- Suggest the smallest effective fix.
- Prefer improving retrieval or rules before suggesting model changes.

OUTPUT RULES
- Return valid JSON only.

OUTPUT SCHEMA
{
  "failure_mode": "missing_context|weak_rule|weak_prompt|bad_severity|duplication|hallucination|other",
  "recommended_fix_type": "retrieval|rule|prompt|reranker|training_data",
  "recommended_fix": "string",
  "candidate_rule_code": "string",
  "candidate_prompt_change": "string"
}

INPUT
REVIEW_OUTPUT
{{review_output}}

HUMAN_FEEDBACK
{{human_feedback}}
```

***

## 9. Deterministic rule prompt template

**Purpose:** when you want the LLM to apply a fixed checklist without open-ended creativity.

```text
ROLE
You are a strict SQL rule checker.

TASK
Evaluate the SQL object only against the supplied checklist rules.

INSTRUCTIONS
- Do not add new rules.
- For each rule, return PASS, FAIL, or NOT_ENOUGH_CONTEXT.
- If FAIL, give one concise evidence statement and one concise fix.
- If the rule does not apply to the object type, return NOT_APPLICABLE.

OUTPUT RULES
- Return valid JSON only.

OUTPUT SCHEMA
{
  "results": [
    {
      "rule_code": "string",
      "status": "PASS|FAIL|NOT_ENOUGH_CONTEXT|NOT_APPLICABLE",
      "evidence": "string",
      "fix": "string"
    }
  ]
}

INPUT
OBJECT_TYPE
{{object_type}}

SQL
{{sql_text}}

CHECKLIST_RULES
{{rules}}
```

***

## 10. Human-readable final review prompt

**Purpose:** turn structured findings into reviewer-friendly prose for UI, Slack, or PR comments.

```text
ROLE
You are a senior reviewer writing concise, high-signal review comments.

TASK
Convert the structured findings into a reviewer-friendly report.

INSTRUCTIONS
- Prioritize critical and high findings first.
- Keep the total comment count small.
- Be direct and specific.
- Each comment must include:
  - issue,
  - why it matters,
  - what to change.
- Do not include praise unless relevant.
- Do not repeat the same point twice.

OUTPUT FORMAT
Return markdown with:
1. Overall risk
2. Top findings
3. Validation queries
4. Deployment checks

INPUT
FINAL_STRUCTURED_REVIEW
{{final_structured_review}}
```

***

## Severity rubric block

Use this same rubric in every review-related prompt to improve consistency. Consistency improves when degrees of freedom are reduced and the model is forced into a fixed rubric with explicit evidence requirements. [skillmaps](https://skillmaps.net/blog/ai-code-reviews-best-practices)

```text
SEVERITY_RUBRIC
- critical: very likely incorrect results, data corruption, dangerous DML, security exposure, or outage risk
- high: strong correctness issue, major performance problem, or serious business-rule mismatch
- medium: meaningful issue that should be fixed soon, but not immediately blocking
- low: useful improvement or preventive cleanup
- info: observation only, no clear immediate action
```

***

## Guardrails block

Use this in all review prompts.

```text
GUARDRAILS
- No finding without evidence.
- No invented schema facts.
- No invented business meaning.
- If uncertain, say uncertain.
- Prefer fewer high-confidence findings over many speculative findings.
- Do not exceed maximum finding count.
- Do not recommend behavior-changing rewrites unless the issue justifies it.
```

***

## Best practices for this prompt pack

This pack is intentionally structured because structured prompts, explicit evidence requirements, and bounded output generally reduce noisy reviews and make downstream automation easier. A two-pass or multi-pass design is also recommended for consistency, since separating extraction from severity assignment and final merge reduces variance and makes false positives easier to suppress. [collinwilkins](https://collinwilkins.com/articles/ai-code-review-best-practices-approaches-tools)

To get better results:
- Keep temperature low for review and summarization.
- Enforce JSON schema validation after each pass.
- Reject findings without evidence.
- Cap findings at 8–10.
- Version prompts in files, not inline strings.
- Feed only relevant context, not the entire repo, because focused context improves signal. [skillmaps](https://skillmaps.net/blog/ai-code-reviews-best-practices)

## Recommended file layout

```text
app/prompts/
  ingestion/
    object_summary.txt
    business_summary.txt
  review/
    technical_review.txt
    business_logic_review.txt
    pr_diff_review.txt
    merge_findings.txt
    validation_queries.txt
    deterministic_rule_check.txt
    final_human_report.txt
  learning/
    feedback_learning.txt
  shared/
    severity_rubric.txt
    guardrails.txt
```

## Suggested runtime order

Use this order in code:
1. deterministic rule check
2. technical review
3. business-logic review
4. merge findings
5. validation-query generation
6. final human-readable report

That order gives you reliable baseline checks first, then richer reasoning, then a cleanup pass before presenting results.

You now have all four parts:
1. PostgreSQL DDL
2. FastAPI skeleton
3. Ingestion pipeline
4. Review prompt pack

A strong next step would be turning all four into a **starter repo structure** with actual files and boilerplate.


