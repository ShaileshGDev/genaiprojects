Here is an extensive, practical guide to prompting for SQL code reviews (jobs, stored procedures, views, table creation, indexing), tailored to your data-engineering and backend background.

## Why prompt quality matters for SQL reviews

SQL code review is high-risk: small mistakes cause performance regressions, locking issues, or data bugs. A good prompt:
- Clearly defines the **goal** (e.g., performance, correctness, maintainability).
- Specifies **context** (RDBMS, schemas, workload, data scale).
- Gives the model a **structured checklist** to follow.
- Asks for **concrete, actionable output** (not just “looks good”).

Bad prompts are vague, context-free, or ask for opinions instead of checks. Excellent prompts are explicit, scoped, and force the model to reason step-by-step. [sqlwithmanoj](https://sqlwithmanoj.com/2015/10/25/code-review-checklist-for-sql-server-stored-procedures-t-sql-scripts/)

***

## Core best practices for writing SQL review prompts

### 1. State the objective and scope explicitly

Include:
- What you want reviewed: job, SP, view, table DDL, index strategy, etc.
- The primary objective: performance, correctness, security, maintainability, schema design, etc.
- The scope: single object, a batch of objects, or a whole module.

Example pattern:
> “Review this SQL Server stored procedure for performance, correctness, and maintainability. Focus on execution plan anti-patterns, transaction usage, error handling, and naming conventions.”

### 2. Provide runtime and workload context

Mention:
- RDBMS: SQL Server, PostgreSQL, T-SQL, etc.
- Data scale: row counts, table sizes, expected growth.
- Workload: OLTP, reporting, ETL, nightly batch, high-concurrency API.
- Any constraints: max latency, locking windows, compliance rules.

Without this, the model can’t judge if an index is “good” or if a join is “acceptable.”

### 3. Define a structured review checklist

Ask the model to evaluate against specific areas, for example:

For **stored procedures / jobs**:
- Parameter usage and data types
- SET options (`SET NOCOUNT ON`, `SET XACT_ABORT ON`)
- Error handling (`TRY…CATCH`, `THROW`)
- Transaction usage scope and duration
- Avoiding `SELECT *`, using explicit columns
- DDL vs DML ordering in the same object
- Use of temp tables vs table variables appropriately
- Anti-patterns: UDFs in joins, correlated subqueries, unnecessary scans
- Index usage hints, missing index signals, scan vs seek patterns

For **views**:
- No `SELECT *`
- Explicit column lists
- Avoiding nested views that hide complexity
- Join types and filter pushdown
- No side effects (triggers, DML in views if possible)

For **table creation DDL**:
- Primary keys and constraints
- `NOT NULL` defaults where appropriate
- Data type choices (precision, scale)
- Partitioning strategy (if relevant)
- Naming conventions and schema placement
- Foreign keys and referential integrity

For **indexing**:
- Primary key indexes
- Covering indexes for frequent queries
- Filtered indexes if supported
- Index fragmentation considerations
- Avoiding over-indexing and redundant indexes

You can embed these as a checklist in the prompt so the model responds section-by-section.

### 4. Ask for step-by-step reasoning and concrete suggestions

Force the model to:
- Walk through the code logically.
- Identify specific lines or patterns.
- Propose refactors with example code.
- Explain *why* something is a problem (e.g., “this causes a scan instead of a seek”).

Example instruction:
> “For each issue, explain the problem, show the problematic pattern, and provide a corrected version with comments.”

### 5. Require prioritization and risk assessment

Ask the model to:
- Categorize issues as: critical, high, medium, low.
- Indicate risk: correctness bug, performance regression, security, maintainability.
- Recommend which items must be fixed before deployment.

This makes the review actionable, not just academic.

### 6. Specify output format

Define how you want the response:
- Sections: “Critical Issues”, “High”, “Medium”, “Low”.
- Per issue: line reference, pattern, explanation, fix example.
- Optional: summary table of issues with severity and impact.

Example:
> “Output as: 1) Critical Issues, 2) High, 3) Medium, 4) Low. Each item: ‘Line X: pattern → issue → fix example’.”

### 7. Include the actual code and relevant metadata

Always:
- Paste the full SQL object (SP, view, DDL, job script).
- Add comments about known hot queries or problematic patterns.
- Optionally include sample query patterns that use the object.

Without the code, the model can only give generic advice.

***

## Example bad prompts

These are weak because they are vague, context-free, or ask for opinions.

### Bad prompt 1: Too vague

> “Review this SQL.”

Problems:
- No objective (performance? correctness?).
- No DB system or context.
- No checklist or output format.
- Likely yields generic, non-actionable advice.

### Bad prompt 2: Opinion ask, no structure

> “Is this stored procedure good? Any problems?”

Problems:
- “Good” is undefined.
- No guidance on what to check.
- No request for line-level feedback or examples.
- Model will likely give superficial comments.

### Bad prompt 3: Ignores context and scale

> “Here’s my view. Can you make it faster?”

Problems:
- No info on data size, query patterns, or workload.
- No DB system specified.
- No constraints (e.g.,不能 change schema).
- Model may suggest unrealistic or harmful changes.

***

## Example good prompts

These are clear, scoped, and structured but not yet maximally powerful.

### Good prompt 1: Stored procedure review

> “Review this SQL Server stored procedure for performance, correctness, and maintainability.  
> Context: OLTP system, tables ~1M rows each, called by a high-concurrency API.  
> Checklist: parameter usage, SET options, error handling, transaction scope, SELECT *, temp tables vs table variables, UDF usage, index usage, anti-patterns.  
> Output: group issues as Critical/High/Medium/Low, with line references, explanation, and corrected code snippets.”

This is good because it:
- States objective and scope.
- Gives workload context.
- Provides a checklist.
-Defines output format.

### Good prompt 2: Table DDL review

> “Review this SQL Server table creation script for schema design, constraints, and indexing strategy.  
> Context: core transactional table, ~5M rows, frequent range queries on `created_at`.  
> Check: primary key, NOT NULL defaults, data types, foreign keys, indexes, naming conventions.  
> Output: list issues by severity, show problematic patterns, and provide revised DDL.”

### Good prompt 3: View review

> “Review this SQL Server view for performance and correctness.  
> Context: used in reporting dashboards, joins 5–7 tables, each ~1–10M rows.  
> Check: SELECT *, column explicitness, join types, filter pushdown, nested views, unnecessary computed columns.  
> Output: critical/high/medium/low categories, with line references, explanation, and a refactored view definition.”

***

## Example excellent prompts

These go further: they impose structure, force reasoning, and tie to your real-world constraints.

### Excellent prompt 1: End-to-end stored procedure review

> “You are an expert SQL Server DBA and code reviewer.  
> Review the following stored procedure for:
> 1) Performance (execution plan anti-patterns, scans vs seeks, index usage, temp table usage)  
> 2) Correctness (transaction boundaries, error handling, NULL handling, data type safety)  
> 3) Maintainability (naming, comments, structure, DDL vs DML ordering)  
>
> Context:
> - RDBMS: SQL Server 2019+
> - Workload: OLTP + nightly ETL
> - Tables: ~10M rows, high write concurrency on core tables
> - This SP is called by a customer-facing API; max allowed latency 200ms under normal load.
>
> Process:
> - Walk through the code line by line.
> - For each issue:
>   - State the line or pattern.
>   - Explain the problem and its impact.
>   - Provide a corrected version with inline comments.
>   - Classify as Critical/High/Medium/Low and note risk type (performance, correctness, security, maintainability).
>
> Output format:
> 1) Critical Issues  
> 2) High  
> 3) Medium  
> 4) Low  
> For each item: “Line X: pattern → issue → fix example (with code)”.
>
> Code:
> ```sql
> -- paste full SP here
> ```

This is excellent because it:
- Defines role and expertise.
- Breaks review into clear dimensions.
- Gives detailed context (latency, data scale).
- Forces line-by-line reasoning with fixes.
- Enforces prioritization and risk classification.
- Specifies exact output structure.

### Excellent prompt 2: Multi-object batch review (SPs + views + tables)

> “You are a senior data platform engineer reviewing database objects for a production sales system.  
> Review these objects:
> - 3 stored procedures (ETL + reporting)
> - 2 views (aggregated dashboards)
> - 1 table creation script + index definitions
>
> Goals:
> - Performance under 5M+ rows, high read/write concurrency
> - Correctness for financial data (no rounding errors, proper NULL handling)
> - Maintainability for a team of 8 developers
>
> For each object:
> 1) Summarize its purpose in 1–2 lines.
> 2) List issues by severity (Critical/High/Medium/Low) with:
>    - Line/pattern
>    - Problem description
>    - Impact (performance, correctness, maintainability)
>    - Refactored code snippet
> 3) Provide a short ‘deployment checklist’ of what must be fixed before release.
>
> RDBMS: SQL Server 2022.  
> Output as separate sections per object, then a final summary table of all issues with severity and impact.
>
> Code:
> ```sql
> -- paste all objects here with comments marking which is SP/view/table
> ```

This is excellent because it:
- Handles multiple objects in one prompt.
- Sets clear business goals (financial correctness, concurrency).
- Requires both per-object detail and a global summary.
- Produces a deployable checklist.

### Excellent prompt 3: Indexing strategy + query pattern review

> “You are an expert SQL Server performance tuner.  
> Review this table DDL and the associated index definitions, plus a list of typical query patterns that use this table.  
>
> Context:
> - Table: ~20M rows, high insert volume, frequent range queries on `created_at` and filtered queries on `status`.
> - Workload: mixed OLTP + reporting.
> - Max acceptable read latency for reporting queries: 1s.
>
> Do:
> 1) Evaluate current indexes:
>    - Are they aligned with key query patterns?
>    - Any missing covering indexes?
>    - Any redundant or overly wide indexes?
> 2) Recommend an index strategy:
>    - Primary key index
>    - Nonclustered indexes for common filters/joins
>    - Filtered indexes if applicable
>    - Any partitioning considerations
> 3) For each recommended index:
>    - Show the CREATE INDEX statement.
>    - Explain which queries it helps and why.
>    - Note any trade-offs (write cost, fragmentation).
>
> Output:
> - Section: ‘Current Index Assessment’
> - Section: ‘Recommended Index Strategy’
> - Section: ‘Query-specific Index Recommendations’ with example queries and index definitions.
>
> DDL and query patterns:
> ```sql
> -- table DDL
> -- sample queries
> ```

This is excellent because it:
- Ties indexes directly to query patterns and latency constraints.
- Requires concrete index DDL and rationale.
- Addresses trade-offs, not just “add more indexes.”

***

## How to adapt these for your environment

Given your stack (PostgreSQL, Python/SQL pipelines, FastAPI, Streamlit):

- Swap “SQL Server” for “PostgreSQL” and adjust checks:
  - Use `RETURNING`, `ON CONFLICT`, `EXPLAIN (ANALYZE, BUFFERS)` instead of SQL Server-specific patterns.
  - Focus on `VACUUM`, extension usage, partitioning, and connection pooling.
- When reviewing ETL jobs:
  - Ask the model to consider batch size, transaction size, and idempotency.
- When reviewing views used by Streamlit:
  - Emphasize materialized views, pre-aggregation, and avoiding expensive joins in real-time queries.

If you want, I can turn one of these templates into a reusable prompt snippet you can drop into your workflow (e.g., as a VS Code snippet or a prompt template for your LLM).

Would you like these examples tailored specifically to PostgreSQL instead of SQL Server?


Here is a reusable, copy-pasteable prompt template you can use either:

- As a **VS Code snippet** (e.g., `sql_review_prompt`)
- Or as a **prompt template** in your LLM tool (Ollama, LM Studio, etc.)

It’s based on the “excellent prompt 1” pattern, but generalized so you can adapt it for SQL Server or PostgreSQL.

***

## Reusable SQL Code Review Prompt Template

```text
**Role:**
You are an expert database engineer and code reviewer specializing in {{DB_TYPE}} (e.g. SQL Server, PostgreSQL).

**Task:**
Review the following SQL object(s) for:
1) Performance (execution plan anti-patterns, scans vs seeks, index usage, temp structures)
2) Correctness (transaction boundaries, error handling, NULL handling, data type safety)
3) Maintainability (naming, comments, structure, clarity, separation of DDL vs DML)

**Context:**
- Database: {{DB_TYPE}}
- Workload: {{WORKLOAD}} (e.g. OLTP, reporting, ETL, nightly batch, high-concurrency API)
- Data scale: {{DATA_SCALE}} (e.g. tables ~1M–10M rows, growth rate, write intensity)
- Performance constraints: {{PERF_CONSTRAINTS}} (e.g. max latency 200ms for API queries, 1s for reporting)
- Business impact: {{BUSINESS_IMPACT}} (e.g. financial data, customer-facing features, internal analytics)

**Objects to review:**
{{OBJECT_TYPE}} (e.g. stored procedure, view, table DDL + indexes, job script)
{{OBJECT_DESCRIPTION}} (brief 1–2 line description of what it does)

**Code:**
{{SQL_CODE}}

**Review process:**
- Walk through the code logically (line by line or by logical block).
- For each issue:
  - Identify the line number or pattern.
  - Explain the problem and its impact (performance, correctness, security, maintainability).
  - Provide a corrected version or refactored snippet with inline comments.
  - Classify severity as: Critical / High / Medium / Low.
  - Indicate risk type: performance, correctness, security, maintainability.

**Checklist by object type (adapt as needed):**

For stored procedures / jobs:
- Parameter usage and data types
- SET options (e.g. NOCOUNT, XACT_ABORT, or PostgreSQL equivalents)
- Error handling (TRY…CATCH / THROW, or EXCEPTION blocks)
- Transaction scope and duration
- Avoiding SELECT *, using explicit columns
- Temp tables vs table variables / temp tables usage patterns
- Anti-patterns: UDFs in joins, correlated subqueries, unnecessary scans
- Index usage hints, missing index signals, scan vs seek patterns

For views:
- No SELECT *; explicit column lists
- Avoiding nested views that hide complexity
- Join types and filter pushdown
- No side effects (triggers, DML in views if possible)

For table creation DDL:
- Primary keys and constraints
- NOT NULL defaults where appropriate
- Data type choices (precision, scale)
- Partitioning strategy (if relevant)
- Naming conventions and schema placement
- Foreign keys and referential integrity

For indexing:
- Primary key indexes
- Covering indexes for frequent queries
- Filtered/partial indexes if supported
- Index fragmentation considerations
- Avoiding over-indexing and redundant indexes

**Output format:**
1) Critical Issues
2) High Issues
3) Medium Issues
4) Low Issues

For each item, use this structure:
- Line X: pattern → issue → severity → risk type → fix example (with code).

After the detailed list, provide:
- A short “Deployment Checklist”: what MUST be fixed before release.
- A short “Post-Release Monitoring Checklist”: what to observe (e.g. query plans, latency, lock waits).

**Conclude with:**
- A 3–5 bullet summary of the most important changes.
- Any major architectural recommendations (e.g. materialized views, partitioning, batching strategy) if applicable.
```

***

## How to use it as a VS Code snippet

1. In VS Code, open:
   - `File` → `Preferences` → `User Snippets` → `json (json)` or create a new snippet file like `sql.json`.
2. Add a snippet entry:

```json
{
  "SQL Review Prompt Template": {
    "prefix": "sqlReviewPrompt",
    "body": [
      "**Role:**",
      "You are an expert database engineer and code reviewer specializing in ${1:DB_TYPE} (e.g. SQL Server, PostgreSQL).",
      "",
      "**Task:**",
      "Review the following SQL object(s) for:",
      "1) Performance (execution plan anti-patterns, scans vs seeks, index usage, temp structures)",
      "2) Correctness (transaction boundaries, error handling, NULL handling, data type safety)",
      "3) Maintainability (naming, comments, structure, clarity, separation of DDL vs DML)",
      "",
      "**Context:**",
      "- Database: ${2:DB_TYPE}",
      "- Workload: ${3:WORKLOAD} (e.g. OLTP, reporting, ETL, nightly batch, high-concurrency API)",
      "- Data scale: ${4:DATA_SCALE} (e.g. tables ~1M–10M rows, growth rate, write intensity)",
      "- Performance constraints: ${5:PERF_CONSTRAINTS} (e.g. max latency 200ms for API queries, 1s for reporting)",
      "- Business impact: ${6:BUSINESS_IMPACT} (e.g. financial data, customer-facing features, internal analytics)",
      "",
      "**Objects to review:**",
      "${7:OBJECT_TYPE} (e.g. stored procedure, view, table DDL + indexes, job script)",
      "${8:OBJECT_DESCRIPTION} (brief 1–2 line description of what it does)",
      "",
      "**Code:**",
      "${9:SQL_CODE}",
      "",
      "**Review process:**",
      "- Walk through the code logically (line by line or by logical block).",
      "- For each issue:",
      "  - Identify the line number or pattern.",
      "  - Explain the problem and its impact (performance, correctness, security, maintainability).",
      "  - Provide a corrected version or refactored snippet with inline comments.",
      "  - Classify severity as: Critical / High / Medium / Low.",
      "  - Indicate risk type: performance, correctness, security, maintainability.",
      "",
      "**Checklist by object type (adapt as needed):**",
      "",
      "For stored procedures / jobs:",
      "- Parameter usage and data types",
      "- SET options (e.g. NOCOUNT, XACT_ABORT, or PostgreSQL equivalents)",
      "- Error handling (TRY…CATCH / THROW, or EXCEPTION blocks)",
      "- Transaction scope and duration",
      "- Avoiding SELECT *, using explicit columns",
      "- Temp tables vs table variables / temp tables usage patterns",
      "- Anti-patterns: UDFs in joins, correlated subqueries, unnecessary scans",
      "- Index usage hints, missing index signals, scan vs seek patterns",
      "",
      "For views:",
      "- No SELECT *; explicit column lists",
      "- Avoiding nested views that hide complexity",
      "- Join types and filter pushdown",
      "- No side effects (triggers, DML in views if possible)",
      "",
      "For table creation DDL:",
      "- Primary keys and constraints",
      "- NOT NULL defaults where appropriate",
      "- Data type choices (precision, scale)",
      "- Partitioning strategy (if relevant)",
      "- Naming conventions and schema placement",
      "- Foreign keys and referential integrity",
      "",
      "For indexing:",
      "- Primary key indexes",
      "- Covering indexes for frequent queries",
      "- Filtered/partial indexes if supported",
      "- Index fragmentation considerations",
      "- Avoiding over-indexing and redundant indexes",
      "",
      "**Output format:**",
      "1) Critical Issues",
      "2) High Issues",
      "3) Medium Issues",
      "4) Low Issues",
      "",
      "For each item, use this structure:",
      "- Line X: pattern → issue → severity → risk type → fix example (with code).",
      "",
      "After the detailed list, provide:",
      "- A short \"Deployment Checklist\": what MUST be fixed before release.",
      "- A short \"Post-Release Monitoring Checklist\": what to observe (e.g. query plans, latency, lock waits).",
      "",
      "**Conclude with:**",
      "- A 3–5 bullet summary of the most important changes.",
      "- Any major architectural recommendations (e.g. materialized views, partitioning, batching strategy) if applicable."
    ],
    "description": "Template prompt for reviewing SQL objects (SPs, views, tables, indexes) with context, checklist, and structured output."
  }
}
```

Now, in any `.sql` file:

- Type `sqlReviewPrompt` and hit Tab.
- VS Code will expand the template with placeholders.
- Fill in:
  - `DB_TYPE`
  - `WORKLOAD`
  - `DATA_SCALE`
  - `PERF_CONSTRAINTS`
  - `OBJECT_TYPE`
  - `OBJECT_DESCRIPTION`
  - `SQL_CODE`

Then send the filled prompt to your LLM.

***

## As a plain prompt template (for LLM UIs)

If you don’t want VS Code snippets, just save the first block (the long “Reusable SQL Code Review Prompt Template”) as a text file, e.g. `sql_review_prompt_template.txt`, and in your LLM UI:

1. Load the template.
2. Replace the `{{...}}` placeholders with your actual values.
3. Paste the full prompt into the chat.

You can adapt `DB_TYPE` and the checklist items per environment (SQL Server vs PostgreSQL) as needed.


Below are exhaustive, structured checklists for all the categories from the first response, adapted for both SQL Server (T‑SQL) and PostgreSQL where relevant. They are written to be:

- Used directly in code reviews.
- Embedded into your prompt templates.
- Turned into team standards or review guidelines.

***

## 1. Stored Procedures & Jobs

### A. Parameters & Data Types

- All parameters explicitly typed (no `@p` without type).
- Parameter types match column types exactly (no implicit conversions).
- No `VARCHAR`/`NVARCHAR` without length; use sensible lengths.
- Use `OPTION (RECOMPILE)` or local variables only when needed for plan stability.
- Avoid `SELECT @var = ...` for single-row assignment unless necessary; prefer `SET`.
- For jobs: parameters are validated before use; no dynamic SQL without parameterization.

### B. SET Options & Session Behavior

**SQL Server:**
- `SET NOCOUNT ON` at procedure start.
- `SET XACT_ABORT ON` to auto-rollback on errors.
- Avoid changing `SET` options that affect plan caching unless necessary.
- Use `SET TRANSACTION_ISOLATION` carefully; document why.

**PostgreSQL:**
- Use `SET` or `LOCAL SET` only where needed; prefer function-level defaults.
- Use `EXCEPTION` blocks consistently.
- Avoid long-lived session-level changes inside functions.

### C. Error Handling

**SQL Server:**
- Every procedure with DML has `TRY…CATCH`.
- In `CATCH`:
  - Log error (table, extended event, or external system).
  - Use `THROW` or `RAISERROR` to propagate meaningful messages.
  - Avoid silent failures; never just `RETURN` without logging.
- Validate input parameters early; return early with clear error messages.

**PostgreSQL:**
- Use `BEGIN … EXCEPTION … END` blocks where failure is expected/handled.
- Log errors via `RAISE LOG/WARNING/ERROR` as appropriate.
- Ensure exceptions don’t leave partial transactions unless intentional.

### D. Transactions

- Transactions are as short as possible (minimize duration).
- Transactions cover only related DML; no long-running user interactions inside.
- Use explicit `BEGIN TRANSACTION` / `COMMIT` / `ROLLBACK` instead of relying on autocommit alone.
- Avoid nested transactions; use `@@TRANCOUNT` (SQL Server) or track logically.
- For jobs:
  - Wrap entire logical job in a transaction if partial success is unacceptable.
  - Use checkpointing for large jobs (save progress, allow restart).

### E. Queries & Column Usage

- No `SELECT *` in procedures/views; always explicit column lists.
- Avoid unnecessary columns in joins and subqueries.
- Use `WHERE` clauses to limit rows early.
- Avoid `DISTINCT` unless semantically required.
- Prefer `JOIN` over correlated subqueries.
- Avoid functions on indexed columns in `WHERE` (e.g., `CAST`, `DATEADD` on key).

### F. Temporary Structures

**SQL Server:**
- Use `#temp` tables for multi-step ETL or complex logic.
- Use table variables only for small, read-only sets.
- Create indexes on temp tables if they are used in joins/aggregations.
- Avoid overusing temp tables for trivial logic.

**PostgreSQL:**
- Use `WITH` (CTE) for readability; be aware of materialization behavior.
- Use temp tables for large ETL steps; add indexes if needed.
- Avoid chaining too many CTEs that hide performance issues.

### G. Anti-Patterns

- No scalar UDFs in `WHERE`/`JOIN` clauses (huge performance cost).
- No `SELECT` into variables inside loops.
- Avoid `OPTION (FORCE ORDER)` unless absolutely necessary.
- Avoid `UNION` where `UNION ALL` is sufficient.
- Avoid `CROSS APPLY` with expensive operations.
- No dynamic SQL without parameterization and validation.
- Avoid `WAITFOR` / sleeping in procedures unless intentional.

### H. Index Usage & Plan Considerations

- Ensure predicates align with index columns (leading columns used).
- Avoid key lookups where possible; consider covering indexes.
- Check for implicit conversions causing scans.
- Use `OPTION (RECOMPILE)` only when parameter sensitivity is critical.
- For jobs:
  - Analyze execution plans for large batch operations.
  - Consider batch size tuning (e.g., chunked deletes/updates).

### I. Naming, Comments & Structure

- Procedure names: `schema_action_object` (e.g., `dbousp_customer_insert`).
- Use consistent naming for parameters: `@p_...` or `p_...`.
- Add header comment: purpose, inputs, outputs, side effects.
- Group code logically: validation → business logic → DML → cleanup.
- Avoid deep nesting; break into multiple procedures if needed.

***

## 2. Views

### A. Column Usage & Explicitness

- No `SELECT *`; always explicit column lists.
- Avoid exposing internal columns that shouldn’t be public.
- Use meaningful column aliases.
- Avoid hiding critical logic in computed columns without documentation.

### B. Joins & Filtering

- Use explicit join types (`JOIN`, `LEFT JOIN`, etc.), not old-style `WHERE` joins.
- Push filters as close to base tables as possible.
- Avoid unnecessary joins that can be removed.
- Ensure join conditions are sargable (no functions on keys).

### C. Complexity & Nesting

- Avoid deeply nested views; limit layers.
- Document complex views with comments.
- Prefer simple views over “mega views” that join 10+ tables.
- Materialize heavy aggregations if needed (materialized views in PostgreSQL, indexed views in SQL Server where appropriate).

### D. Side Effects & Constraints

- No DML in views (unless intentional and documented).
- Avoid triggers that fire on view access.
- Ensure views don’t cause unexpected row multiplication.

### E. Performance & Optimization

- Use covering indexes on base tables referenced by views.
- Avoid `ORDER BY` in views unless required (can be misleading).
- For reporting views:
  - Consider pre-aggregated tables instead of complex views.
  - Evaluate冰箱: materialized views vs repeated computation.

### F. Naming & Documentation

- View names: `schema_view_object` (e.g., `salesview_customer_summary`).
- Add header comments: purpose, key tables, business rules.
- Document any non-obvious transformations.

***

## 3. Table Creation & Schema Design

### A. Keys & Constraints

- Every table has a primary key.
- Use appropriate key types:
  - Surrogate keys (`INT`, `BIGINT`, identity, or UUID) for transactional tables.
  - Natural keys only when truly stable and unique.
- Define foreign keys for referential integrity.
- Use `UNIQUE` constraints for business uniqueness.
- Avoid relying solely on application logic for constraints.

### B. Data Types

- Choose appropriate types:
  - `INT`/`BIGINT` for IDs.
  - `DECIMAL`/`NUMERIC` for monetary values.
  - `DATE`/`TIMESTAMP` for temporal data.
- Avoid `VARCHAR` without length; use reasonable lengths.
- Use `NOT NULL` with sensible defaults where possible.
- Avoid `NVARCHAR` unless Unicode is required.

### C. Defaults & NULL Handling

- Use `DEFAULT` values for common cases (e.g., `created_at`, `status`).
- Document when `NULL` is allowed and what it means.
- Avoid “magic NULLs” (e.g., `NULL` as “unknown” without documentation).

### D. Indexing Strategy

- Primary key automatically has an index.
- Add indexes on:
  - Foreign key columns.
  - Columns frequently used in `WHERE`, `JOIN`, `ORDER BY`.
- Use covering indexes for frequent query patterns.
- Avoid over-indexing:
  - Measure write impact.
  - Remove unused indexes.

### E. Partitioning & Scaling

- Consider partitioning for:
  - Very large tables (e.g., time-series, logs).
  - Tables with clear range keys (`created_at`, `order_date`).
- Document partition strategy (range, list, hash).
- Ensure queries can benefit from partition pruning.

### F. Naming & Conventions

- Table names: `schema_entity` (e.g., `salescustomer`, `ordersorder`).
- Use lowercase or consistent casing per team standard.
- Avoid reserved words and special characters.
- Use consistent naming for:
  - IDs: `customer_id`, `order_id`.
  - Timestamps: `created_at`, `updated_at`.

### G. Security & Compliance

- Sensitive columns:
  - Consider encryption at rest or column-level encryption.
  - Document access restrictions.
- Avoid storing unnecessary PII.
- Use roles and schemas to enforce access boundaries.

***

## 4. Indexing

### A. Primary & Unique Indexes

- Every table has a primary key with an index.
- Use unique constraints for business uniqueness.
- Ensure primary key is narrow and stable.

### B. Nonclustered & Covering Indexes

- Add indexes on:
  - Foreign keys.
  - Frequently filtered columns.
  - Join columns.
- Create covering indexes for high-frequency queries:
  - Include all columns used in `SELECT`, `WHERE`, `JOIN`, `ORDER BY`.
- Avoid duplicate indexes with same columns and order.

### C. Filtered / Partial Indexes

**SQL Server:**
- Use filtered indexes for subsets (e.g., `WHERE status = 'ACTIVE'`).

**PostgreSQL:**
- Use partial indexes: `CREATE INDEX ... WHERE status = 'ACTIVE'`.

Ensure queries match the filter condition exactly.

### D. Index Maintenance & Fragmentation

- Monitor fragmentation periodically.
- Rebuild/reorganize indexes as needed.
- Avoid excessive index changes during peak hours.
- Consider index maintenance jobs (nightly/weekly).

### E. Index Design Rules

- Leading columns in index should match most common filter patterns.
- Avoid indexes on low-cardinality columns unless used in specific patterns.
- Avoid wide indexes (too many columns); favor focused indexes.
- Ensure statistics are up-to-date for accurate plans.

### F. Anti-Patterns

- No indexes on columns never used in predicates.
- No indexes that cause massive write overhead without benefit.
- No redundant indexes with different names but same columns.
- Avoid indexing computed columns unless heavily used.

***

## 5. ETL Jobs & Batch Operations

### A. Job Design & Control

- Jobs have clear entry/exit points and logging.
- Support restartability:
  - Save progress (e.g., last processed ID, timestamp).
  - Allow resume from checkpoint.
- Use configuration tables or parameters for job behavior.

### B. Transaction & Batch Size

- Break large operations into batches:
  - E.g., `DELETE/UPDATE` in chunks of 10k–100k rows.
- Use transactions per batch to limit lock duration.
- Avoid single transaction over millions of rows.

### C. Error Handling & Logging

- Log:
  - Start/end times.
  - Rows processed.
  - Errors and warnings.
- Use dedicated log tables or external systems.
- Implement retry logic for transient errors.

### D. Data Quality & Validation

- Validate input data before loading:
  - Type checks, range checks, referential checks.
- Use staging tables for transformation.
- Compare source vs target counts; report discrepancies.

### E. Performance & Resource Usage

- Schedule jobs during low-usage windows if possible.
- Monitor:
  - CPU, I/O, lock waits.
  - Temp DB usage.
- Tune batch sizes based on observed performance.

### F. Security & Access

- Jobs run with minimal necessary permissions.
- Avoid embedded credentials; use role-based access.
- Log job access and changes to job definitions.

***

## 6. Jobs (Scheduled/Background)

### A. Configuration & Parameters

- Job parameters are centralized and documented.
- Use environment-specific configs (dev/test/prod).
- Avoid hardcoded values in job logic.

### B. Monitoring & Alerts

- Jobs emit status: success/failure/partial.
- Alerts on:
  - Failure.
  - Excessive duration.
  - Unexpected row counts.
- Include links to logs or dashboards.

### C. Idempotency & Safety

- Jobs are idempotent where possible:
  - Re-running doesn’t duplicate data.
- Use upsert logic (`MERGE`, `INSERT … ON CONFLICT`).
- Document side effects clearly.

### D. Cleanup & Retention

- Jobs clean up intermediate data (temp tables, staging).
- Define retention policies for logs and history.
- Archive old data instead of deleting if needed.

***

## 7. General SQL Script Quality

### A. Readability

- Consistent indentation (e.g., 2–4 spaces).
- One logical clause per line (`SELECT`, `FROM`, `WHERE`, etc.).
- Use meaningful aliases (`c` for customer, `o` for order).
- Avoid one-liners with nested logic.

### B. Comments

- Header comment for each object:
  - Purpose.
  - Inputs/outputs.
  - Side effects.
- Inline comments for complex logic.
- Avoid over-commenting trivial code.

### C. Consistency

- Consistent naming across objects.
- Consistent patterns for:
  - Error handling.
  - Transactions.
  - Logging.
- Follow team style guide.

### D. Versioning & Change Management

- Scripts are versioned (Git).
- Use migration scripts instead of ad-hoc changes.
- Document changes in changelog or commit messages.
- Avoid direct manual changes on production.

### E. Testing & Validation

- Each object has:
  - Unit tests (where applicable).
  - Integration tests for critical paths.
- Validate:
  - Row counts.
  - Data correctness.
  - Performance under realistic load.

***

These checklists can be:

- Copied into your review documentation.
- Used as part of the “Checklist by object type” section in your prompt template.
- Adapted into automated linting rules (if you build tools around them).

If you want, I can next compress these into a single “SQL Review Checklist” document tailored specifically to PostgreSQL + your ETL/FastAPI/Streamlit stack.

*****

Below is a single, consolidated **SQL Review Checklist** tailored specifically to **PostgreSQL** and your stack: **data pipelines, ETL, FastAPI backends, and Streamlit dashboards**. You can:

- Drop this into your team’s review docs.
- Use it as the “Checklist by object type” section inside your prompt template.
- Turn individual sections into automated linting or validation rules later.

***

# PostgreSQL SQL Review Checklist  
*(For Data Pipelines, ETL, FastAPI, Streamlit)*

## 1. Stored Procedures & Functions (PL/pgSQL)

### 1.1 Parameters & Data Types
- All parameters explicitly typed (`INTEGER`, `TEXT`, `TIMESTAMP`, etc.).
- Parameter types match column types exactly (no implicit casts in predicates).
- Use `TEXT`/`VARCHAR` with care; avoid overly long `VARCHAR` limits.
- Avoid passing raw SQL strings as parameters unless necessary and validated.

### 1.2 Session & Transaction Behavior
- Use `BEGIN … EXCEPTION … END` consistently for error handling.
- Prefer short transactions; avoid long-running user interactions.
- For critical ETL:
  - Wrap logical steps in explicit transactions.
  - Use checkpointing (save progress) for restartability.

### 1.3 Error Handling & Logging
- Use `RAISE LOG/WARNING/ERROR` appropriately:
  - `LOG`: informational.
  - `WARNING`: potential issues.
  - `ERROR`: abort current transaction.
- Ensure errors are not silently swallowed; always log or propagate.
- For ETL functions:
  - Log row counts, start/end times, and key metrics.
  - Record failures in a dedicated log table.

### 1.4 Queries & Column Usage
- No `SELECT *`; always explicit column lists.
- Avoid unnecessary columns in joins and subqueries.
- Use `WHERE` clauses to limit rows early.
- Avoid `DISTINCT` unless semantically required.
- Prefer `JOIN` over correlated subqueries.
- Avoid functions on indexed columns in `WHERE` (e.g., `CAST`, `DATE_TRUNC` on key).

### 1.5 Temporary Structures & CTEs
- Use `WITH` (CTE) for readability; be aware of materialization behavior.
- For large ETL steps:
  - Use temp tables (`CREATE TEMP TABLE`) instead of long CTE chains.
  - Add indexes on temp tables if used in joins/aggregations.
- Avoid overusing CTEs that hide performance issues.

### 1.6 Anti-Patterns
- No scalar functions in `WHERE`/`JOIN` clauses that cause row-by-row execution.
- Avoid `SELECT` into variables inside loops.
- Avoid `UNION` where `UNION ALL` is sufficient.
- No dynamic SQL without parameterization and validation.
- Avoid `pg_sleep` or sleeping in functions unless intentional.

### 1.7 Index Usage & Plan Considerations
- Ensure predicates align with index columns (leading columns used).
- Avoid key lookups where possible; consider covering indexes.
- Check for implicit conversions causing scans.
- Use `EXPLAIN (ANALYZE, BUFFERS)` to validate performance.
- For large batch operations:
  - Analyze execution plans.
  - Tune batch sizes (e.g., chunked deletes/updates).

### 1.8 Naming, Comments & Structure
- Function names: `schema_action_object` (e.g., `sales.fn_insert_customer`).
- Use consistent naming for parameters: `p_...`.
- Add header comment: purpose, inputs, outputs, side effects.
- Group code logically: validation → business logic → DML → cleanup.
- Avoid deep nesting; break into multiple functions if needed.

***

## 2. Views (Regular & Materialized)

### 2.1 Column Usage & Explicitness
- No `SELECT *`; always explicit column lists.
- Avoid exposing internal columns that shouldn’t be public.
- Use meaningful column aliases (`customer_name` instead of `c.name`).
- Document non-obvious transformations in comments.

### 2.2 Joins & Filtering
- Use explicit join types (`JOIN`, `LEFT JOIN`, etc.).
- Push filters as close to base tables as possible.
- Avoid unnecessary joins that can be removed.
- Ensure join conditions are sargable (no functions on keys).

### 2.3 Complexity & Nesting
- Avoid deeply nested views; limit layers.
- Document complex views with comments.
- Prefer simple views over “mega views” that join 10+ tables.
- For heavy aggregations:
  - Prefer materialized views or pre-aggregated tables.
  - Use `REFRESH MATERIALIZED VIEW` in ETL jobs.

### 2.4 Side Effects & Constraints
- No DML in views (unless intentional and documented).
- Avoid triggers that fire on view access unless necessary.
- Ensure views don’t cause unexpected row multiplication.

### 2.5 Performance & Optimization
- Use covering indexes on base tables referenced by views.
- Avoid `ORDER BY` in views unless required (can be misleading).
- For reporting views used by Streamlit:
  - Consider pre-aggregated tables instead of complex views.
  - Evaluate materialized views vs repeated computation.

### 2.6 Naming & Documentation
- View names: `schema_view_object` (e.g., `salesview_customer_summary`).
- Add header comments: purpose, key tables, business rules.
- Document any non-obvious transformations.

***

## 3. Table Creation & Schema Design

### 3.1 Keys & Constraints
- Every table has a primary key.
- Use appropriate key types:
  - `BIGINT` with `GENERATED`/`IDENTITY` for IDs.
  - `UUID` when needed for distributed systems.
- Define foreign keys for referential integrity.
- Use `UNIQUE` constraints for business uniqueness.
- Avoid relying solely on application logic for constraints.

### 3.2 Data Types
- Choose appropriate types:
  - `INT`/`BIGINT` for IDs.
  - `NUMERIC` for monetary values (avoid `FLOAT` for money).
  - `DATE`/`TIMESTAMP`/`TIMESTAMPTZ` for temporal data.
- Avoid `VARCHAR` without length; use reasonable lengths.
- Use `NOT NULL` with sensible defaults where possible.
- Avoid unnecessary `JSON`/`JSONB` unless truly flexible schema is needed.

### 3.3 Defaults & NULL Handling
- Use `DEFAULT` values for common cases (`created_at`, `status`).
- Document when `NULL` is allowed and what it means.
- Avoid “magic NULLs” (e.g., `NULL` as “unknown” without documentation).

### 3.4 Indexing Strategy
- Primary key automatically has an index.
- Add indexes on:
  - Foreign key columns.
  - Columns frequently used in `WHERE`, `JOIN`, `ORDER BY`.
- Use covering indexes for frequent query patterns.
- Avoid over-indexing:
  - Measure write impact.
  - Remove unused indexes.

### 3.5 Partitioning & Scaling
- Consider partitioning for:
  - Very large tables (e.g., time-series, logs).
  - Tables with clear range keys (`created_at`, `order_date`).
- Document partition strategy (range, list).
- Ensure queries can benefit from partition pruning.

### 3.6 Naming & Conventions
- Table names: `schema_entity` (e.g., `salescustomer`, `ordersorder`).
- Use lowercase or consistent casing per team standard.
- Avoid reserved words and special characters.
- Use consistent naming for:
  - IDs: `customer_id`, `order_id`.
  - Timestamps: `created_at`, `updated_at`.

### 3.7 Security & Compliance
- Sensitive columns:
  - Consider encryption at rest or column-level encryption.
  - Document access restrictions.
- Avoid storing unnecessary PII.
- Use roles and schemas to enforce access boundaries.

***

## 4. Indexing

### 4.1 Primary & Unique Indexes
- Every table has a primary key with an index.
- Use unique constraints for business uniqueness.
- Ensure primary key is narrow and stable.

### 4.2 Nonclustered & Covering Indexes
- Add indexes on:
  - Foreign keys.
  - Frequently filtered columns.
  - Join columns.
- Create covering indexes for high-frequency queries:
  - Include all columns used in `SELECT`, `WHERE`, `JOIN`, `ORDER BY`.
- Avoid duplicate indexes with same columns and order.

### 4.3 Partial (Filtered) Indexes
- Use partial indexes for subsets:
  ```sql
  CREATE INDEX idx_orders_active
  ON orders (created_at)
  WHERE status = 'ACTIVE';
  ```
- Ensure queries match the filter condition exactly.

### 4.4 Index Maintenance & Fragmentation
- Monitor bloat and fragmentation with `pg_stat_user_indexes`, `pgstattuple`.
- Reindex periodically for large, heavily updated tables.
- Avoid excessive index changes during peak hours.
- Consider index maintenance jobs (nightly/weekly).

### 4.5 Index Design Rules
- Leading columns in index should match most common filter patterns.
- Avoid indexes on low-cardinality columns unless used in specific patterns.
- Avoid wide indexes (too many columns); favor focused indexes.
- Ensure statistics are up-to-date (`ANALYZE`) for accurate plans.

### 4.6 Anti-Patterns
- No indexes on columns never used in predicates.
- No indexes that cause massive write overhead without benefit.
- No redundant indexes with different names but same columns.
- Avoid indexing computed columns unless heavily used.

***

## 5. ETL Jobs & Batch Operations

### 5.1 Job Design & Control
- Jobs have clear entry/exit points and logging.
- Support restartability:
  - Save progress (e.g., last processed ID, timestamp).
  - Allow resume from checkpoint.
- Use configuration tables or parameters for job behavior.

### 5.2 Transaction & Batch Size
- Break large operations into batches:
  - E.g., `DELETE/UPDATE` in chunks of 10k–100k rows.
- Use transactions per batch to limit lock duration.
- Avoid single transaction over millions of rows.

### 5.3 Error Handling & Logging
- Log:
  - Start/end times.
  - Rows processed.
  - Errors and warnings.
- Use dedicated log tables or external systems.
- Implement retry logic for transient errors.

### 5.4 Data Quality & Validation
- Validate input data before loading:
  - Type checks, range checks, referential checks.
- Use staging tables for transformation.
- Compare source vs target counts; report discrepancies.

### 5.5 Performance & Resource Usage
- Schedule jobs during low-usage windows if possible.
- Monitor:
  - CPU, I/O, lock waits.
  - Temp tablespace usage.
- Tune batch sizes based on observed performance.

### 5.6 Security & Access
- Jobs run with minimal necessary permissions.
- Avoid embedded credentials; use role-based access.
- Log job access and changes to job definitions.

***

## 6. Background / Scheduled Jobs (e.g., pg_cron, Airflow)

### 6.1 Configuration & Parameters
- Job parameters are centralized and documented.
- Use environment-specific configs (dev/test/prod).
- Avoid hardcoded values in job logic.

### 6.2 Monitoring & Alerts
- Jobs emit status: success/failure/partial.
- Alerts on:
  - Failure.
  - Excessive duration.
  - Unexpected row counts.
- Include links to logs or dashboards.

### 6.3 Idempotency & Safety
- Jobs are idempotent where possible:
  - Re-running doesn’t duplicate data.
- Use upsert logic (`INSERT … ON CONFLICT`).
- Document side effects clearly.

### 6.4 Cleanup & Retention
- Jobs clean up intermediate data (temp tables, staging).
- Define retention policies for logs and history.
- Archive old data instead of deleting if needed.

***

## 7. General SQL Script Quality

### 7.1 Readability
- Consistent indentation (2–4 spaces).
- One logical clause per line (`SELECT`, `FROM`, `WHERE`, etc.).
- Use meaningful aliases (`c` for customer, `o` for order).
- Avoid one-liners with nested logic.

### 7.2 Comments
- Header comment for each object:
  - Purpose.
  - Inputs/outputs.
  - Side effects.
- Inline comments for complex logic.
- Avoid over-commenting trivial code.

### 7.3 Consistency
- Consistent naming across objects.
- Consistent patterns for:
  - Error handling.
  - Transactions.
  - Logging.
- Follow team style guide.

### 7.4 Versioning & Change Management
- Scripts are versioned (Git).
- Use migration scripts instead of ad-hoc changes.
- Document changes in changelog or commit messages.
- Avoid direct manual changes on production.

### 7.5 Testing & Validation
- Each object has:
  - Unit tests (where applicable).
  - Integration tests for critical paths.
- Validate:
  - Row counts.
  - Data correctness.
  - Performance under realistic load.

***

You can now:

- Paste this checklist into your team’s review guidelines.
- Embed its sections into your prompt template as the “Checklist by object type”.
- Use it as a manual review rubric before merging any SQL changes.
