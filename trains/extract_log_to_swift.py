#!/usr/bin/env python3
"""
Extract GSA training samples from the proxy-log sqlite database.

Reads requests.sqlite3 (read-only), resolves media paths, parses responses
containing bbox/point/affordance JSON, and converts to ms-swift format.

Outputs:
  - trains/gsa_log_sft.jsonl          (SFT: system+user+assistant with images)
  - trains/gsa_log_testset/<date>.jsonl (per-day test splits for benchmark)
  - trains/log_extract_report.json     (statistics)

Usage:
    python trains/extract_log_to_swift.py [--limit 0] [--test_days_ratio 0.1]
    python trains/extract_log_to_swift.py --limit 1000  # small test run
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOG_ROOT = Path("/mnt/data/datasets/qwen-proxy-log-mirror-fast")
DB_PATH = LOG_ROOT / "data/index/requests.sqlite3"
HOT_DIR = LOG_ROOT / "data/hot"
ARCHIVE_DIR = LOG_ROOT / "data/archive"
MEDIA_DIR = LOG_ROOT / "data/hot/media"

SYSTEM_PROMPT = (
    "You are an embodied perception assistant specialized in indoor scene understanding. "
    "You can detect objects, understand spatial relationships, identify object attributes and states, "
    "and reason about robot manipulation affordances. Respond precisely with structured output."
)

# Patterns to detect GSA-relevant content in responses
BOX_PATTERN = re.compile(r'"bbox"\s*:\s*\[')
POINT_PATTERN = re.compile(r'"affordance_point"\s*:\s*\[')
TAG_BOX_PATTERN = re.compile(r'<box>')
TAG_POINT_PATTERN = re.compile(r'<point>')


def resolve_media_path(sha_ref: str) -> Optional[str]:
    """Resolve sha256:xxx reference to actual file path."""
    if not sha_ref:
        return None
    sha = sha_ref.split(":", 1)[1] if sha_ref.startswith("sha256:") else sha_ref
    prefix = sha[:2]
    for kind in ["images", "videos"]:
        d = MEDIA_DIR / kind / prefix
        if d.exists():
            for p in d.glob(f"{sha}.*"):
                return str(p)
    return None


def find_detail_dir(detail_path: str) -> Optional[Path]:
    """Find the detail directory in hot or archive."""
    if not detail_path:
        return None
    for base in [HOT_DIR, ARCHIVE_DIR]:
        candidate = base / detail_path
        if candidate.exists():
            return candidate
    return None


def parse_response_content(content: str) -> Optional[Dict]:
    """Parse response content for GSA-relevant structured data."""
    if not content:
        return None
    if content.strip() in ("<FAILED>", ""):
        return None

    # Try parsing as JSON (most common format for GSA responses)
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            has_bbox = "bbox" in data
            has_point = "affordance_point" in data or "point" in data
            if has_bbox or has_point:
                return data
    except json.JSONDecodeError:
        pass

    # Check for <box>/<point> tag format
    if TAG_BOX_PATTERN.search(content) or TAG_POINT_PATTERN.search(content):
        return {"raw_tagged": content}

    # Check for JSON-like bbox/point patterns in free text
    if BOX_PATTERN.search(content) or POINT_PATTERN.search(content):
        return {"raw_json_like": content}

    return None


def classify_dimension(parsed: Dict) -> str:
    """Classify the dimension based on parsed response content."""
    if "raw_tagged" in parsed:
        content = parsed["raw_tagged"]
        if "<box>" in content:
            return "G_grounding"
        if "<point>" in content:
            return "G_grounding"
        return "G_grounding"

    if "affordance_point" in parsed:
        if "preferred_approach" in parsed or "grasp_hint" in parsed:
            return "A_action_interface"
        return "G_grounding"
    if "bbox" in parsed:
        if "object_state" in parsed or "state" in parsed:
            return "A_obj_state"
        return "G_grounding"
    if "spatial" in str(parsed).lower():
        return "S_spatial"
    return "G_grounding"


def extract_user_text_from_request(raw_request: dict) -> Optional[str]:
    """Extract user text content from raw_request messages."""
    messages = raw_request.get("messages", [])
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            return "\n".join(texts)
    return None


def build_swift_record(user_text: str, assistant_content: str,
                       image_path: str, dimension: str) -> dict:
    """Build ms-swift format record."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\n{user_text}"},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [image_path],
        "dimension": dimension,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract GSA samples from proxy log")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of rows to process (0=all)")
    parser.add_argument("--test_days_ratio", type=float, default=0.1,
                        help="Fraction of days to hold out as test set")
    parser.add_argument("--min_completion_tokens", type=int, default=30,
                        help="Skip responses with fewer tokens")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir or str(script_dir)
    testset_dir = os.path.join(output_dir, "gsa_log_testset")
    os.makedirs(testset_dir, exist_ok=True)

    print(f"DB: {args.db}")
    print(f"Output: {output_dir}")
    print(f"Limit: {'all' if args.limit == 0 else args.limit}")
    print()

    # Connect read-only
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Query: requests with media and reasonable completion
    query = """
        SELECT req_id, ts, detail_path, media_refs_json, model, completion_tokens
        FROM requests
        WHERE media_refs_json IS NOT NULL AND media_refs_json != '[]'
          AND completion_tokens >= ?
        ORDER BY ts ASC
    """
    params = [args.min_completion_tokens]
    if args.limit > 0:
        query += " LIMIT ?"
        params.append(args.limit)

    print("Querying database...")
    t0 = time.time()
    rows = conn.execute(query, params).fetchall()
    print(f"  {len(rows)} rows fetched in {time.time()-t0:.1f}s")

    # Process rows
    sft_records = []
    by_date = defaultdict(list)
    stats = {
        "total_rows": len(rows),
        "detail_found": 0,
        "detail_missing": 0,
        "media_found": 0,
        "media_missing": 0,
        "response_parsed": 0,
        "response_failed": 0,
        "response_empty": 0,
        "records_produced": 0,
        "by_dimension": defaultdict(int),
        "by_model": defaultdict(int),
    }

    print("Processing rows...")
    for i, row in enumerate(rows):
        if i > 0 and i % 10000 == 0:
            print(f"  {i}/{len(rows)} processed, {stats['records_produced']} records so far")

        detail_path = row["detail_path"]
        media_refs = json.loads(row["media_refs_json"] or "[]")

        # Resolve media
        if not media_refs:
            stats["media_missing"] += 1
            continue
        image_path = resolve_media_path(media_refs[0])
        if not image_path:
            stats["media_missing"] += 1
            continue
        stats["media_found"] += 1

        # Find detail dir
        detail_dir = find_detail_dir(detail_path)
        if not detail_dir:
            stats["detail_missing"] += 1
            continue
        stats["detail_found"] += 1

        # Read raw_response
        resp_file = detail_dir / "raw_response.json"
        stream_file = detail_dir / "stream.jsonl"
        response_content = None

        if resp_file.exists():
            try:
                resp_data = json.loads(resp_file.read_text())
                choices = resp_data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    response_content = msg.get("content", "")
            except (json.JSONDecodeError, KeyError):
                pass
        elif stream_file.exists():
            # Reconstruct from stream chunks
            try:
                chunks = []
                for line in stream_file.read_text().splitlines():
                    if line.strip():
                        chunk = json.loads(line)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            chunks.append(delta["content"])
                response_content = "".join(chunks)
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

        if not response_content:
            stats["response_empty"] += 1
            continue

        # Parse response for GSA content
        parsed = parse_response_content(response_content)
        if not parsed:
            stats["response_failed"] += 1
            continue
        stats["response_parsed"] += 1

        # Read raw_request to get user text
        req_file = detail_dir / "raw_request.json"
        user_text = None
        if req_file.exists():
            try:
                req_data = json.loads(req_file.read_text())
                user_text = extract_user_text_from_request(req_data)
            except json.JSONDecodeError:
                pass

        if not user_text:
            user_text = "Analyze this image and provide structured perception output."

        # Classify dimension
        dimension = classify_dimension(parsed)

        # Build record
        record = build_swift_record(user_text, response_content, image_path, dimension)

        # Extract date for test split
        ts = row["ts"] or ""
        date_str = ts[:10]  # YYYY-MM-DD

        sft_records.append(record)
        by_date[date_str].append(record)
        stats["records_produced"] += 1
        stats["by_dimension"][dimension] += 1
        stats["by_model"][row["model"] or "unknown"] += 1

    conn.close()

    # Split test set by days
    all_dates = sorted(by_date.keys())
    n_test_days = max(1, int(len(all_dates) * args.test_days_ratio))
    test_dates = set(random.sample(all_dates, min(n_test_days, len(all_dates))))

    train_records = []
    test_records_by_date = {}

    for date_str, recs in by_date.items():
        if date_str in test_dates:
            test_records_by_date[date_str] = recs
        else:
            train_records.extend(recs)

    random.shuffle(train_records)

    # Write outputs
    sft_path = os.path.join(output_dir, "gsa_log_sft.jsonl")
    with open(sft_path, "w") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n  Train: {len(train_records)} records -> {sft_path}")

    for date_str, recs in sorted(test_records_by_date.items()):
        test_path = os.path.join(testset_dir, f"{date_str}.jsonl")
        with open(test_path, "w") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Test [{date_str}]: {len(recs)} records")

    total_test = sum(len(v) for v in test_records_by_date.values())
    print(f"  Test total: {total_test} records across {len(test_records_by_date)} days")

    # Report
    stats["by_dimension"] = dict(stats["by_dimension"])
    stats["by_model"] = dict(stats["by_model"])
    stats["train_records"] = len(train_records)
    stats["test_records"] = total_test
    stats["test_dates"] = sorted(test_dates)

    report_path = os.path.join(output_dir, "log_extract_report.json")
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {report_path}")

    print(f"\n{'='*60}")
    print(f"  Total rows:     {stats['total_rows']}")
    print(f"  Records:        {stats['records_produced']}")
    print(f"  Train:          {len(train_records)}")
    print(f"  Test:           {total_test}")
    print(f"  Detail found:   {stats['detail_found']}")
    print(f"  Detail missing: {stats['detail_missing']}")
    print(f"  Media found:    {stats['media_found']}")
    print(f"  Media missing:  {stats['media_missing']}")
    print(f"  Parse success:  {stats['response_parsed']}")
    print(f"  Parse fail:     {stats['response_failed']}")
    for dim, cnt in sorted(stats["by_dimension"].items()):
        print(f"    {dim}: {cnt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
