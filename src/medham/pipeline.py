"""
pipeline.py
Async orchestrator — collects LLM responses under all 4 experimental conditions.

STUDY DESIGN (2×2 Factorial)
  Factor A — Prompt format : zero_shot (ZS) vs citation-required (CIT)
                              Operationalises H2.3 (citation prompting effect)
  Factor B — Context       : no RAG (NoRAG) vs retrieval-augmented (RAG)
                              Operationalises H2.1 (RAG hallucination reduction)

  4 conditions × 4 tested models = 16 cells per benchmark item.

  Models tested (is_tested=1):
    GPT-4o              — commercial frontier (OpenAI)        H2.2: frontier tier
    Claude Sonnet 4.6   — commercial frontier (Anthropic)     H2.2: frontier tier
    Gemini 2.5 Flash    — commercial frontier (Google)        H2.2: frontier tier
    Llama 3.1 8B Inst.  — open-source, quantised via Groq     H2.2: open-source tier

  DeepSeek V4 Flash is NOT called here — reserved as independent judge and FActScore
  decomposer in labeler.py (eliminates self-evaluation bias; Zheng et al., 2023).

REPRODUCIBILITY PARAMETERS — document all of these in the methodology
  GENERATION_TEMPERATURE = 0.0  (see constant below)
    Why: temperature=0 produces near-deterministic outputs. Standard for
    reproducible medical AI evaluation (Menz et al., 2024; Hager et al., 2024).
    Non-zero temperature introduces stochasticity that prevents exact replication.

  MAX_TOKENS = 600
    Why: sufficient for a complete medical answer (<400 tokens typical).
    Capped to control cost and prevent citation list padding in CIT conditions.
    NOTE: response_token_count is logged per response so the length confound
    (citation-prompted responses are structurally longer) can be controlled for
    in the mixed-effects logistic regression.

  Model version pinning:
    API model strings do NOT uniquely identify frozen model versions — providers
    silently update underlying weights. The responded_at timestamp bounds the
    model version. Document the date range in the methodology:
    "Report the actual response collection date range from responded_at values."
    When available, use snapshot identifiers (e.g. "gpt-4o-2024-11-20").

  Llama 3.1 8B via Groq — quantisation caveat (report in methodology):
    "llama-3.1-8b-instant" is Groq's quantised deployment, not Meta's fp16 weights.
    Results reflect Groq's inference infrastructure. The H2.2 frontier vs
    open-source comparison therefore conflates three factors: (1) model scale,
    (2) RLHF/alignment maturity, (3) quantisation. This must be acknowledged as
    a limitation — the effect cannot be attributed to any single factor alone.

ENGINEERING DESIGN
  Async-First, Sort-Then-Insert:
    Phase 1 — All API calls fire concurrently. Responses staged to disk
               immediately — crash-safe at any point in the run.
    Phase 2 — Staged rows sorted (item_id, strategy_id, model_id ASC)
               and bulk-inserted into model_responses. This guarantees
               deterministic response_id ordering across all runs.

  Per-provider semaphores: each API's rate limits respected independently.
  Resumable: skips (item, model, strategy) already in model_responses/staging.
  Retry: exponential backoff on 429/500/502/503/504 — transient failures.

Run order:
  schema.py → pipeline.py → labeler.py → analysis.ipynb
"""

import os
import sys
import time
import asyncio
import sqlite3
import textwrap
from datetime import datetime, timezone
from dotenv import load_dotenv

# Force UTF-8 stdout on Windows — prevents cp1252 UnicodeEncodeError when
# model responses contain non-ASCII characters (e.g. ≥, μ, α, →).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "medham.db")

# Reproducibility constants — each has a methodological justification.
MAX_TOKENS = 600
# Why 600: sufficient for a complete medical answer. Capped to prevent citation
# list padding inflating response length in CIT conditions (length confound).

GENERATION_TEMPERATURE = 0.0
# Why 0.0: near-deterministic output — identical prompt → identical response
# (modulo GPU floating-point non-determinism). Standard for reproducible medical
# AI evaluation. References: Menz et al. (2024), Hager et al. (2024).
# Any non-zero temperature makes exact replication of individual responses
# impossible, even with the same model and prompt.

# Per-provider concurrency limits (stay within free/paid tier rate limits)
SEMAPHORE_LIMITS = {
    "OpenAI":    8,
    "Anthropic": 5,
    "Google":    8,
    "Groq":      1,  # free tier: 30 RPM — serialise + sleep to avoid 429s
}

MAX_RETRIES = 4
RETRY_BASE  = 2.0   # seconds — doubles each retry (2, 4, 8, 16)

# Refusal signal strings — matched against lowercased responses before staging.
# Pattern-only: no word-count heuristic. A short but medically valid answer
# ("Aspirin 81mg daily for thromboprophylaxis.") must never be flagged as a refusal.
# Patterns are specific enough to avoid false positives on real clinical answers.
_REFUSAL_SIGNALS = (
    "i cannot provide",          "i can't provide",
    "i am unable to",            "i'm unable to",
    "i cannot help",             "i can't help",
    "as an ai, i",               "as a language model",
    "i must advise you to consult",
    "please consult a",          "please seek medical",
    "i don't have the ability to",
    "i recommend consulting",    "i would recommend seeing",
    "this is beyond my",         "i cannot give medical",
)

def build_clients() -> dict:
    from openai import OpenAI
    import anthropic
    from google import genai

    return {
        "openai": OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        ),
        "anthropic": anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        ),
        "google": genai.Client(
            api_key=os.environ["GOOGLE_API_KEY"]
        ),
        "groq": OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"]
        ),
    }

def _call_model_sync(clients: dict, model_version: str, provider: str,
                     prompt: str) -> tuple[str, int, int, int]:
    """
    Dispatches to the correct API based on provider.
    Returns (response_text, tokens_input, tokens_output, response_time_ms).
    Raises exception on failure — caller handles retry/log.
    """
    if provider == "OpenAI":
        t0   = time.perf_counter()
        resp = clients["openai"].chat.completions.create(
            model=model_version,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
        )
        ms = round((time.perf_counter() - t0) * 1000)
        return (
            resp.choices[0].message.content.strip(),
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
            ms,
        )

    elif provider == "Anthropic":
        t0   = time.perf_counter()
        resp = clients["anthropic"].messages.create(
            model=model_version,
            max_tokens=MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        ms   = round((time.perf_counter() - t0) * 1000)
        text = resp.content[0].text.strip() if resp.content else ""
        return (
            text,
            resp.usage.input_tokens,
            resp.usage.output_tokens,
            ms,
        )

    elif provider == "Google":
        from google.genai import types
        t0   = time.perf_counter()
        resp = clients["google"].models.generate_content(
            model=model_version,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_TOKENS,
                temperature=GENERATION_TEMPERATURE,
            ),
        )
        ms = round((time.perf_counter() - t0) * 1000)
        if not resp.candidates:
            raise ValueError("Gemini returned no candidates — possible quota or safety block")
        candidate     = resp.candidates[0]
        finish_reason = str(getattr(candidate, "finish_reason", "")).upper()
        if finish_reason in ("SAFETY", "RECITATION", "BLOCKED"):
            raise ValueError(f"Gemini response blocked by API: finish_reason={finish_reason}")
        try:
            text = resp.text.strip()
        except Exception:
            if candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text.strip()
            else:
                raise ValueError("Gemini: cannot extract text from response")
        in_tok  = getattr(resp.usage_metadata, "prompt_token_count",     0) or 0
        out_tok = getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
        return (text, in_tok, out_tok, ms)

    elif provider == "Groq":
        t0   = time.perf_counter()
        resp = clients["groq"].chat.completions.create(
            model=model_version,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
        )
        ms = round((time.perf_counter() - t0) * 1000)
        return (
            resp.choices[0].message.content.strip(),
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
            ms,
        )

    else:
        raise ValueError(f"Unknown provider: {provider!r}")

async def call_model_async(
    clients:       dict,
    model_version: str,
    provider:      str,
    prompt:        str,
    semaphore:     asyncio.Semaphore,
) -> tuple[str, int, int]:
    """Async wrapper with semaphore and exponential-backoff retry on rate limits."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                result = await asyncio.to_thread(
                    _call_model_sync, clients, model_version, provider, prompt
                )
                # Groq free tier: pace calls to stay under 30 RPM.
                # 7.0s ≈ 8.5 RPM — well below the 30 RPM limit.
                # Sleep while semaphore is still held so no other coroutine
                # fires immediately after a successful call.
                if provider == "Groq":
                    await asyncio.sleep(7.0)
                return result
            except Exception as e:
                err_str      = str(e).lower()
                is_rate_limit = (
                    "429" in str(e)
                    or "rate limit" in err_str
                    or "too many"   in err_str
                    or "quota"      in err_str
                )
                # Retry on rate-limit errors AND transient server errors.
                # 429: rate limit. 500/502/503/504: transient infrastructure
                # failures that resolve on retry without changing the request.
                is_transient = (
                    is_rate_limit
                    or "500" in str(e)
                    or "502" in str(e)
                    or "503" in str(e)
                    or "504" in str(e)
                    or "resource_exhausted"      in err_str
                    or "service unavailable"     in err_str
                    or "bad gateway"             in err_str
                    or "gateway timeout"         in err_str
                    or "temporarily unavailable" in err_str
                )
                if is_transient and attempt < MAX_RETRIES - 1:
                    if is_rate_limit and provider == "Groq":
                        # Groq 60s rolling window — wait long enough to clear it.
                        wait = 62.0
                    else:
                        wait = RETRY_BASE ** (attempt + 1)
                    print(f"\n  [RETRY] {provider} — transient error, waiting {wait:.0f}s "
                          f"(attempt {attempt + 1}/{MAX_RETRIES}): {str(e)[:300]}")
                    await asyncio.sleep(wait)
                else:
                    raise

def build_prompt(template: str, question: str, context: str = "") -> str:
    return template.replace("{question}", question).replace("{context}", context)

# STAGING TABLE — crash-safe buffer before sorted commit

def create_staging_table(conn: sqlite3.Connection) -> None:
    """
    Creates model_responses_staging as a crash-safe buffer for async results.
    Uses CREATE TABLE IF NOT EXISTS so a table from a previous crashed
    run is preserved — those staged rows will be picked up by the resume
    check and included in the final sorted commit.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_responses_staging (
            staging_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id          INTEGER NOT NULL,
            model_id         INTEGER NOT NULL,
            strategy_id      INTEGER NOT NULL,
            context_id       INTEGER,
            response_text    TEXT,
            tokens_input     INTEGER,
            tokens_output    INTEGER,
            response_time_ms INTEGER,
            responded_at     TEXT    NOT NULL,
            UNIQUE(item_id, model_id, strategy_id)
        );
    """)
    conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM model_responses_staging;")
    n = cur.fetchone()[0]
    if n > 0:
        print(f"  NOTE: Found {n} rows in model_responses_staging from a previous run — will merge.")
    else:
        print("  model_responses_staging created (empty).")

def commit_staging(conn: sqlite3.Connection) -> None:
    """
    Reads model_responses_staging ordered by (item_id, strategy_id, model_id)
    and bulk-inserts into model_responses in that exact order.

    Because SQLite AUTOINCREMENT assigns IDs strictly in insertion order,
    this guarantees deterministic response_id ordering:
      response_id=1 → item_id=1, strategy_id=1, model_id=1
      response_id=2 → item_id=1, strategy_id=1, model_id=2
      ...
    After a successful commit, model_responses_staging is dropped.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT item_id, model_id, strategy_id, context_id,
               response_text, tokens_input, tokens_output, response_time_ms, responded_at
        FROM   model_responses_staging
        ORDER  BY item_id ASC, strategy_id ASC, model_id ASC
    """)
    staged_rows = cur.fetchall()

    if not staged_rows:
        print("  No staged rows to commit.")
        conn.execute("DROP TABLE IF EXISTS model_responses_staging;")
        conn.commit()
        return

    print(f"\n  -- Phase 2: Committing {len(staged_rows)} staged rows in sorted order --")

    inserted = 0
    updated  = 0
    skipped  = 0
    for row in staged_rows:
        i_id, m_id, s_id, ctx_id, r_text, t_in, t_out, t_ms, r_at = row
        # First try to overwrite an existing ERROR row with the new result.
        # UPDATE only fires if a prior ERROR row exists for this (item, model, strategy).
        upd = conn.execute("""
            UPDATE model_responses
            SET    response_text=?, tokens_input=?, tokens_output=?,
                   response_time_ms=?, responded_at=?
            WHERE  item_id=? AND model_id=? AND strategy_id=?
              AND  response_text LIKE 'ERROR:%'
        """, (r_text, t_in, t_out, t_ms, r_at, i_id, m_id, s_id))
        if upd.rowcount > 0:
            updated += 1
            continue
        # No ERROR row — insert fresh (IGNORE if a successful row already exists).
        ins = conn.execute("""
            INSERT OR IGNORE INTO model_responses
                (item_id, model_id, strategy_id, context_id,
                 response_text, tokens_input, tokens_output, response_time_ms, responded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (i_id, m_id, s_id, ctx_id, r_text, t_in, t_out, t_ms, r_at))
        if ins.rowcount > 0:
            inserted += 1
        else:
            skipped += 1

    conn.commit()
    print(f"  Committed: {inserted} new | {updated} retried (ERROR->OK) | {skipped} skipped (already OK)")

    conn.execute("DROP TABLE IF EXISTS model_responses_staging;")
    conn.commit()
    print("  model_responses_staging dropped. Sorted insertion complete.")

# ASYNC MAIN PIPELINE

async def run_pipeline(conn: sqlite3.Connection, clients: dict) -> None:
    cur = conn.cursor()

    # Load only is_tested=1 models — pipeline does not call judge-only models
    cur.execute("""
        SELECT model_id, model_name, model_version, provider
        FROM   models
        WHERE  is_tested = 1
        ORDER  BY model_id;
    """)
    models = cur.fetchall()

    cur.execute("SELECT item_id, question FROM benchmark_items ORDER BY item_id;")
    questions = cur.fetchall()

    cur.execute("""
        SELECT strategy_id, strategy_name, prompt_type, rag_enabled, prompt_template
        FROM   prompt_strategies;
    """)
    strategies = cur.fetchall()

    cur.execute("SELECT item_id, context_id, knowledge_context FROM rag_contexts;")
    rag_map = {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    semaphores = {p: asyncio.Semaphore(SEMAPHORE_LIMITS[p]) for p in SEMAPHORE_LIMITS}

    total_calls = len(questions) * len(models) * len(strategies)
    counter     = {"completed": 0, "failed": 0, "skipped": 0}
    start_time  = time.time()
    lock        = asyncio.Lock()

    print(f"\n  Benchmark items : {len(questions)}")
    print(f"  Tested models   : {len(models)}")
    print(f"  Strategies      : {len(strategies)}")
    print(f"  Total API calls : {total_calls}")
    print(f"  Phase 1: Async collection -> model_responses_staging")
    print("  " + "-" * 70)

    async def process_one(item_id, question, model_id, model_name,
                          model_version, provider, strategy_id, strategy_name,
                          rag_enabled, template):

        # Skip only if a SUCCESSFUL response already exists.
        # ERROR rows are retried — committed as placeholders after all retries
        # failed, not as valid responses. Staging rows are always retried too.
        async with lock:
            cur2 = conn.cursor()
            cur2.execute("""
                SELECT 1 FROM model_responses
                WHERE item_id=? AND model_id=? AND strategy_id=?
                  AND response_text NOT LIKE 'ERROR:%'
                UNION ALL
                SELECT 1 FROM model_responses_staging
                WHERE item_id=? AND model_id=? AND strategy_id=?
                LIMIT 1
            """, (item_id, model_id, strategy_id,
                  item_id, model_id, strategy_id))
            if cur2.fetchone():
                counter["skipped"] += 1
                return

        context_id, rag_text = rag_map.get(item_id, (None, ""))
        context = rag_text if rag_enabled else ""
        prompt  = build_prompt(template, question, context)
        ctx_id  = context_id if rag_enabled else None

        response_text = None
        tokens_in     = None
        tokens_out    = None
        latency_ms    = None
        call_ok       = False

        try:
            sem = semaphores[provider]
            response_text, tokens_in, tokens_out, latency_ms = await call_model_async(
                clients, model_version, provider, prompt, sem
            )
            call_ok = True

        except Exception as e:
            response_text = f"ERROR: {str(e)[:300]}"

        # Refusal detection — flag before staging so labeler can report separately.
        # Models occasionally refuse medical questions ("I cannot provide specific
        # medical advice"). Refusals are NOT hallucinations — they must be excluded
        # from the hallucination rate denominator and reported as a separate count.
        # Prefixing with "REFUSAL:" preserves the text for qualitative analysis
        # while flagging it for exclusion from quantitative evaluation.
        if call_ok and response_text:
            rt_lower = response_text.lower()
            if any(sig in rt_lower for sig in _REFUSAL_SIGNALS):
                response_text = "REFUSAL: " + response_text

        # Stage response immediately (crash-safe)
        now = datetime.now(timezone.utc).isoformat()
        async with lock:
            conn.execute("""
                INSERT OR IGNORE INTO model_responses_staging
                    (item_id, model_id, strategy_id, context_id,
                     response_text, tokens_input, tokens_output, response_time_ms, responded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item_id, model_id, strategy_id, ctx_id,
                response_text, tokens_in, tokens_out, latency_ms, now
            ))
            conn.commit()

            if call_ok:
                counter["completed"] += 1
            else:
                counter["failed"] += 1

            done      = counter["completed"] + counter["failed"] + counter["skipped"]
            pct       = done / total_calls * 100
            elapsed   = time.time() - start_time
            rate      = done / elapsed if elapsed > 0 else 0
            remaining = (total_calls - done) / rate if rate > 0 else 0

            preview = textwrap.shorten(response_text or "", width=55, placeholder="...")
            print(
                f"  [{done:>4}/{total_calls}] {pct:>5.1f}% | "
                f"item={item_id:<3} {model_name:<20} {strategy_name:<10} | "
                f"ETA {remaining / 60:.1f}m | {preview}"
            )

    # Create all coroutines — only over is_tested models
    tasks = []
    for item_id, question in questions:
        for model_id, model_name, model_version, provider in models:
            for strategy_id, strategy_name, _, rag_enabled, template in strategies:
                tasks.append(process_one(
                    item_id, question,
                    model_id, model_name, model_version, provider,
                    strategy_id, strategy_name, rag_enabled, template
                ))

    await asyncio.gather(*tasks)

    # Phase 2: commit staging → model_responses in sorted order
    commit_staging(conn)

    elapsed_min = (time.time() - start_time) / 60
    cur.execute("SELECT COUNT(*) FROM model_responses WHERE response_text NOT LIKE 'ERROR:%';")
    total_stored = cur.fetchone()[0]

    print("\n  " + "=" * 70)
    print(f"  Pipeline complete in {elapsed_min:.1f} minutes")
    print(f"  Completed  : {counter['completed']}")
    print(f"  Failed     : {counter['failed']}")
    print(f"  Skipped    : {counter['skipped']} (already done)")
    print(f"  Total valid responses in DB: {total_stored}")

    print("\n  Responses per model:")
    cur.execute("""
        SELECT m.model_name, COUNT(*) as cnt
        FROM   model_responses r
        JOIN   models m ON r.model_id = m.model_id
        WHERE  r.response_text NOT LIKE 'ERROR:%'
        GROUP  BY m.model_name
        ORDER  BY m.model_id;
    """)
    for row in cur.fetchall():
        print(f"    {row[0]:<25} {row[1]:>4} responses")

    if counter["failed"] > 0:
        print(f"\n  WARNING: {counter['failed']} calls failed. Re-run pipeline.py to retry.")
        print("  (Successful calls are already saved — only failed ones will retry.)")
    else:
        print("\n  All calls succeeded. Ready to run labeler.py")

# ENTRY POINT

if __name__ == "__main__":
    print(f"\n{'=' * 72}")
    print("  MedHallu Evaluation Pipeline (Async-First, Sort-Then-Insert)")
    print(f"{'=' * 72}\n")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite FK enforcement is per-connection
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM benchmark_items;")
    n_q = cur.fetchone()[0]
    if n_q == 0:
        print("ERROR: No benchmark items found. Run schema.py first.")
        conn.close()
        sys.exit(1)
    # Infer whether schema.py was run in TEST_MODE or FULL mode from item count.
    # (TEST_MODE is set in schema.py; pipeline.py reads the result from the DB.)
    mode_note = "PILOT run" if n_q <= 20 else "FULL dataset run"
    print(f"  Benchmark items : {n_q}  [{mode_note}]")
    print(f"  IMPORTANT: TEST_MODE is set in schema.py — re-run schema.py to change mode.")

    cur.execute("SELECT COUNT(*) FROM models WHERE is_tested=1;")
    n_tested = cur.fetchone()[0]
    print(f"  Found {n_tested} tested models in database.")

    required = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
                "GROQ_API_KEY"]
    missing  = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"ERROR: Missing API keys: {missing}")
        conn.close()
        sys.exit(1)

    print("  All API keys found.\n")

    try:
        print("  Building API clients...")
        clients = build_clients()
        print("  Creating staging table...")
        create_staging_table(conn)
        print("\n  Starting async API calls (Phase 1)...\n")
        asyncio.run(run_pipeline(conn, clients))
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user. All staged responses saved — re-run to continue.")
    finally:
        conn.close()
