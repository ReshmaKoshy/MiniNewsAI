# vertex_gemini_single.py
# -*- coding: utf-8 -*-
# Concurrent, non-batched labeling with Vertex AI (Gemini 2.5 Pro) using direct REST calls.
# - Splits input across N threads (default = 18)
# - Each thread processes a chunk and writes:
#     vertex_single_runs/<thread_id>_ids_<min>-<max>/ip_thread<thread_id>_ids_<min>-<max>.csv
#     vertex_single_runs/<thread_id>_ids_<min>-<max>/token_usage_log_thread<thread_id>_ids_<min>-<max>.csv
# - Per-thread pacing only: we sleep to meet a min interval (default 7.0s)
# - Final merged CSV: vertex_single_runs/df_with_labels.csv

import os
import re
import json
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from math import floor
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from dotenv import load_dotenv

# ----------------------------
# Config
# ----------------------------
load_dotenv()

PROJECT_ID   = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION     = os.environ.get("GOOGLE_CLOUD_LOCATION")
MODEL_ID     = "gemini-2.5-pro"
ACCESS_TOKEN = os.environ["GEMINI_API_KEY"]

INPUT_CSV_PATH = Path("scripts/input_dataset2.csv")

WORKDIR = Path("./vertex_single_runs")
WORKDIR.mkdir(parents=True, exist_ok=True)

FINAL_OUT_CSV_PATH = WORKDIR / "df_with_labels.csv"

# Generation config
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", 1000))
TEMPERATURE       = float(os.environ.get("TEMPERATURE", 0.2))

# Concurrency & pacing (simple)
THREADS                = int(os.environ.get("THREADS", 18))         # default 18 as per your plan
PER_THREAD_MIN_INTERVAL_SEC = float(os.environ.get("PER_THREAD_MIN_INTERVAL_SEC", 7.0))
# If each inference is guaranteed to take >= 7s, this interval is conservative and fine.

# Optional: snapshot every N rows (set to 0 to disable)
SNAPSHOT_EVERY = int(os.environ.get("SNAPSHOT_EVERY", 0))

SYSTEM_PROMPT = """You are an ethical children’s(aged 8-10) news editor. For each incoming text (news article, summary, excerpt, etc.) do the following, in order:

1. CLASSIFY into exactly one of:
   • SAFE[confidence score] — “Safe”
     Content has no mature, violent, sexual, or discriminatory themes.
     Dont append a parenthetical reason.
   • SENSITIVE[confidence score]”
     Content involves social issues (e.g. mental health, fairness, historical injustice), mild conflict or upset, or themes parents might first want to explain.
     Optionally append a brief parenthetical reason, e.g. “(contains mental health).”
   • UNSAFE[confidence score]”
     Content involves explicit violence, adult sexuality, mature identity issues (LGBTQ+ in an adult context), political extremism, or highly disturbing material.
     Optionally append a brief parenthetical reason, e.g. “(contains explicit violence).”

2. APPLY PARENTAL JUDGMENT:
   – If most parents wouldn’t let their 8–10 year-olds see the source material directly, mark “Unsafe.”
   – If it raises real-world empathy or fairness lessons but has complexity or emotional weight, mark “Sensitive.”
   – Otherwise mark “Safe.”
"""

# ----------------------------
# Regex helpers
# ----------------------------
LABEL_RE = re.compile(r'\b(SAFE|SENSITIVE|UNSAFE)\b', re.IGNORECASE)
CONF_RE  = re.compile(r'\b(SAFE|SENSITIVE|UNSAFE)\s*\[\s*([0-9]*\.?[0-9]+)\s*%?\s*\]', re.IGNORECASE)

def extract_label_confidence(full_text: str) -> Tuple[Optional[str], float]:
    if not full_text:
        return (None, float("nan"))
    m = CONF_RE.search(full_text)
    if m:
        try:
            return (m.group(1).upper(), float(m.group(2)))
        except Exception:
            return (m.group(1).upper(), float("nan"))
    m2 = LABEL_RE.search(full_text)
    if m2:
        return (m2.group(1).upper(), float("nan"))
    return (None, float("nan"))

# ----------------------------
# REST helpers
# ----------------------------
def build_url() -> str:
    return (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"
    )

def call_generate_content(text: str, session: requests.Session, token: str) -> Dict[str, Any]:
    url = build_url()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": str(text)}]},
        ],
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
        },
    }
    resp = session.post(url, headers=headers, data=json.dumps(body), timeout=120)
    resp.raise_for_status()
    return resp.json()

def extract_text_from_response(data: Dict[str, Any]) -> str:
    text = ""
    try:
        cands = data.get("candidates", [])
        if cands:
            parts = cands[0].get("content", {}).get("parts", [])
            for p in parts:
                if isinstance(p, dict) and "text" in p:
                    text += p["text"]
    except Exception:
        pass
    return text

# ----------------------------
# Worker
# ----------------------------
def process_chunk(thread_id: int,
                  df_chunk: pd.DataFrame,
                  snapshot_every: int = 0) -> Path:
    """
    Process a slice of the dataframe on this thread. Writes to its own folder & files.
    Returns the path to the per-thread CSV.
    """
    # Determine the ID range label for paths
    if "id" in df_chunk.columns and not df_chunk["id"].empty:
        min_id = int(df_chunk["id"].min())
        max_id = int(df_chunk["id"].max())
    else:
        # fallback to dataframe index range if 'id' not present
        min_id = int(df_chunk.index.min())
        max_id = int(df_chunk.index.max())

    range_label = f"{min_id}-{max_id}"

    # Thread-local paths (include thread id and id range)
    tdir = WORKDIR / f"{thread_id}_ids_{range_label}"
    tdir.mkdir(parents=True, exist_ok=True)

    out_csv_path = tdir / f"wip_ip_thread{thread_id}_ids_{range_label}.csv"
    log_csv_path = tdir / f"token_usage_log_thread{thread_id}_ids_{range_label}.csv"

    # Ensure output columns exist
    for col in ("label", "reasoning", "confidence"):
        if col not in df_chunk.columns:
            df_chunk[col] = pd.NA

    session = requests.Session()
    min_interval = max(PER_THREAD_MIN_INTERVAL_SEC, 0.0)
    last_call_time = 0.0

    # init log with header
    if not log_csv_path.exists():
        with open(log_csv_path, "w", encoding="utf-8") as f:
            f.write("row_index,prompt_tokens,output_tokens,total_tokens,finish_reason,elapsed_seconds,http_error\n")

    for i, row in df_chunk.iterrows():
        text = str(row["summary_500"])
        start_req = time.time()

        # per-thread pacing ONLY (simple)
        since_last = start_req - last_call_time
        if since_last < min_interval:
            time.sleep(min_interval - since_last)
 
        elapsed = None
        finish_reason = "N/A"
        prompt_tokens = output_tokens = total_tokens = None
        http_error = ""

        delay = 2
        max_retries = 1#3

        for attempt in range(max_retries):
            try:
                req_start = time.time()
                data = call_generate_content(text, session, ACCESS_TOKEN)
                elapsed = time.time() - req_start

                meta = data.get("usageMetadata", {})
                prompt_tokens = meta.get("promptTokenCount")
                output_tokens = meta.get("candidatesTokenCount")
                total_tokens = meta.get("totalTokenCount")
                finish_reason = data.get("candidates", [{}])[0].get("finishReason", "N/A")

                full_text = extract_text_from_response(data)
                label, conf = extract_label_confidence(full_text)

                df_chunk.at[i, "reasoning"]  = full_text
                df_chunk.at[i, "label"]      = label
                df_chunk.at[i, "confidence"] = conf
                break  # success

            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                http_error = f"HTTPError:{status}"
                # Retry on 429 and 5xx
                # if status in (429,) or (status is not None and 500 <= status < 600):
                #     time.sleep(delay)
                #     delay = min(delay * 2, 30)
                #     continue
                # otherwise record and stop
                df_chunk.at[i, "reasoning"]  = f"ERROR: HTTP {status}: {e}"
                df_chunk.at[i, "label"]      = pd.NA
                df_chunk.at[i, "confidence"] = pd.NA
                break

            # except (requests.ConnectionError, requests.Timeout) as e:
            #     http_error = f"{type(e).__name__}"
            #     time.sleep(delay)
            #     delay = min(delay * 2, 30)
            #     if attempt == max_retries - 1:
            #         df_chunk.at[i, "reasoning"]  = f"ERROR: {type(e).__name__}: {e}"
            #         df_chunk.at[i, "label"]      = pd.NA
            #         df_chunk.at[i, "confidence"] = pd.NA
            # except Exception as e:
            #     http_error = f"{type(e).__name__}"
            #     df_chunk.at[i, "reasoning"]  = f"ERROR: {type(e).__name__}: {e}"
            #     df_chunk.at[i, "label"]      = pd.NA
            #     df_chunk.at[i, "confidence"] = pd.NA
            #     break
            finally:
                last_call_time = time.time()

        with open(log_csv_path, "a", encoding="utf-8") as f:
            f.write(f"{i},{prompt_tokens},{output_tokens},{total_tokens},{finish_reason},{(elapsed or 0):.2f},{http_error}\n")

        # write per-row progress (no index to simplify merge)
        df_chunk.to_csv(out_csv_path, index=False)

        # if snapshot_every and ((len(df_chunk.loc[:i]) % snapshot_every) == 0):
        #     snap = tdir / f"df_snapshot_row_{i}.csv"
        #     df_chunk.loc[:i].to_csv(snap, index=False)

    return out_csv_path

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    if not INPUT_CSV_PATH.exists():
        raise SystemExit(f"Input CSV not found: {INPUT_CSV_PATH}")

    df = pd.read_csv(INPUT_CSV_PATH)

    if "summary_500" not in df.columns:
        raise SystemExit("Input CSV must include a 'summary_500' column.")

    columns_to_keep = ["id", "category", "summary_500"]
    df = df[columns_to_keep]

    # Preserve original order for the final merge
    df["_orig_idx"] = range(len(df))

    # Split into THREADS (almost) evenly
    threads = max(1, THREADS)
    chunks: list[pd.DataFrame] = []
    if threads <= 1 or len(df) <= threads:
        chunks = [df]
    else:
        sizes = [len(df) // threads + (1 if x < len(df) % threads else 0) for x in range(threads)]
        start = 0
        for size in sizes:
            if size <= 0:
                continue
            end = start + size
            chunks.append(df.iloc[start:end].copy())
            start = end

    # Run workers
    per_thread_csvs: list[Path] = []
    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        futures = {
            ex.submit(process_chunk, tid, chunk, SNAPSHOT_EVERY): tid
            for tid, chunk in enumerate(chunks, start=1)
        }
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                path = fut.result()
                print(f"[thread {tid}] done -> {path}")
                per_thread_csvs.append(path)
            except Exception as e:
                print(f"[thread {tid}] ERROR: {e}")

    # Merge all per-thread CSVs
    parts = []
    for p in per_thread_csvs:
        part = pd.read_csv(p)
        parts.append(part)

    if not parts:
        raise SystemExit("No per-thread outputs found — nothing to merge.")

    merged = pd.concat(parts, axis=0, ignore_index=True)

    # restore original order on processed subset (if present)
    if "_orig_idx" in merged.columns:
        merged = merged.sort_values(by="_orig_idx").reset_index(drop=True)
        merged = merged.drop(columns=["_orig_idx"])

    FINAL_OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(FINAL_OUT_CSV_PATH, index=False)
    print(f"Done. Wrote: {FINAL_OUT_CSV_PATH}")