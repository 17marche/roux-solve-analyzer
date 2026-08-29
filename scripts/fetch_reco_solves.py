#!/usr/bin/env python3
"""Ingestion script for downloading, caching, and validating reco.nz Roux solves dataset."""

from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from typing import Dict, Any, Optional, List, Tuple

from roux_engine.data.loader import RecoDatasetLoader, SolveRecord
from roux_engine.core.parser import MoveParser


def parse_solvestats(html: str) -> Optional[Dict[str, Any]]:
    """Extracts step timings and STM move splits from the #solvestats table."""
    table_m = re.search(r'<table id=[\"\']solvestats[\"\']>(.*?)</table>', html, re.DOTALL)
    if not table_m:
        return None

    rows = re.findall(r'<tr>(.*?)</tr>', table_m.group(1), re.DOTALL)
    if not rows:
        return None

    headers = [re.sub(r'<[^>]+>', '', th).strip().lower() for th in re.findall(r'<th>(.*?)</th>', rows[0], re.DOTALL)]
    col_names = [h for h in headers if h]

    splits: Dict[str, Any] = {}
    for r in rows[1:]:
        header_cell = re.search(r'<th>(.*?)</th>', r, re.DOTALL)
        if not header_cell:
            continue
        metric = re.sub(r'<[^>]+>', '', header_cell.group(1)).strip().lower()
        tds = [re.sub(r'<[^>]+>', '', td).strip() for td in re.findall(r'<td>(.*?)</td>', r, re.DOTALL)]
        if len(tds) == len(col_names):
            metric_dict: Dict[str, Any] = {}
            for col, val in zip(col_names, tds):
                try:
                    if '.' in val:
                        metric_dict[col] = float(val)
                    elif val.isdigit():
                        metric_dict[col] = int(val)
                    else:
                        metric_dict[col] = val
                except ValueError:
                    metric_dict[col] = val
            splits[metric] = metric_dict
    return splits if splits else None


def extract_solve_from_html(html: str, solve_id: int) -> Tuple[str, str, Optional[str], Optional[Dict[str, Any]]]:
    """Extracts (scramble, solution, raw_reconstruction, human_splits) from raw solve HTML."""
    scramble = ""
    solution = ""
    raw_recon: Optional[str] = None

    # 1. Look for alg.cubing.net link
    cubing_m = re.search(r'href=[\"\']https://alg\.cubing\.net/\?setup=([^&\"\'\s]+)&alg=([^&\"\'\s]+)[\"\']', html)
    if cubing_m:
        scramble = urllib.parse.unquote_plus(cubing_m.group(1)).strip()
        solution = urllib.parse.unquote_plus(cubing_m.group(2)).strip()

    # 2. Extract raw reconstruction text
    recon_m = re.search(r'<div id=[\"\']reconstruction[\"\']>(.*?)</div>', html, re.DOTALL)
    if recon_m:
        raw_html = recon_m.group(1)
        raw_text = raw_html.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
        raw_recon = re.sub(r'<[^>]+>', '', raw_text).strip()

        # If alg.cubing.net wasn't found, try parsing from raw reconstruction text
        if not scramble and raw_recon:
            lines = [l.strip() for l in raw_recon.splitlines() if l.strip()]
            if lines:
                scramble = lines[0]
                solution = "\n".join(lines[1:])

    # 3. Extract human split timings table
    human_splits = parse_solvestats(html)

    return scramble, solution, raw_recon, human_splits


def fetch_or_load_html(solve_id: int, cache_dir: str, retries: int = 3, timeout: int = 12) -> Optional[str]:
    """Loads HTML from disk cache or fetches from reco.nz with retries."""
    cache_path = os.path.join(cache_dir, f"{solve_id}.html")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass

    url = f"https://reco.nz/solve/{solve_id}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RouxCoach/1.0"}
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read().decode("utf-8", errors="replace")
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(content)
                time.sleep(0.05)  # Polite throttle
                return content
        except Exception as e:
            if attempt == retries - 1:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def process_solve_entry(
    item: Dict[str, Any],
    cache_dir: str,
    loader: RecoDatasetLoader
) -> Dict[str, Any]:
    """Fetches, parses, and validates a single solve entry."""
    solve_id = int(item["id"])
    html = fetch_or_load_html(solve_id, cache_dir)
    
    if not html:
        return {
            **item,
            "scramble": "",
            "solution": "",
            "raw_reconstruction": None,
            "human_splits": None,
            "is_valid": False,
            "validation_status": "UNSOLVABLE",
            "invalid_reason": "Failed to fetch HTML page from reco.nz"
        }

    scramble, solution, raw_recon, human_splits = extract_solve_from_html(html, solve_id)

    if not scramble or not solution:
        is_valid, status, reason = False, "UNSOLVABLE", "Missing scramble or solution"
    else:
        is_valid, status, reason = loader.validate_solve_detailed(scramble, solution, allow_rotations=True)

    tps_val = float(item["tps"]) if item.get("tps") else None

    record = SolveRecord(
        id=solve_id,
        scramble=scramble,
        solution=solution,
        solver=item.get("solver"),
        puzzle=item.get("puzzle", "3x3"),
        result=item.get("result"),
        competition=item.get("competition"),
        date=item.get("date"),
        tps=tps_val,
        is_valid=is_valid,
        validation_status=status,
        invalid_reason=reason,
        raw_reconstruction=raw_recon,
        human_splits=human_splits
    )

    d = record.to_dict()
    # Preserve extra metadata from index
    if "method" in item:
        d["method"] = item["method"]
    if "tags" in item:
        d["tags"] = item["tags"]
    if "reconstructed_by" in item:
        d["reconstructed_by"] = item["reconstructed_by"]
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch, cache, and validate reco.nz solves dataset.")
    parser.add_argument("--index", default=None, help="Path to input index JSON (defaults to data/roux_solves.json or data/roux_solves_index.json)")
    parser.add_argument("--output", default="data/roux_solves.json", help="Path to output solves dataset JSON")
    parser.add_argument("--cache-dir", default="data/.cache_reco", help="Path to disk cache directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent download workers")
    parser.add_argument("--limit", type=int, default=None, help="Optional max solves to process")
    args = parser.parse_args()

    index_path = args.index
    if index_path is None:
        if os.path.exists("data/roux_solves_index.json"):
            index_path = "data/roux_solves_index.json"
        elif os.path.exists("data/roux_solves.json"):
            index_path = "data/roux_solves.json"
        else:
            print("Error: No index file found at data/roux_solves.json or data/roux_solves_index.json.", file=sys.stderr)
            sys.exit(1)

    if not os.path.exists(index_path):
        print(f"Error: Index file '{index_path}' not found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    if args.limit:
        index_data = index_data[:args.limit]

    total = len(index_data)
    print(f"Starting ingestion of {total} solves (workers={args.workers}, cache='{args.cache_dir}')...")

    loader = RecoDatasetLoader()
    results: List[Dict[str, Any]] = []
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_item = {
            executor.submit(process_solve_entry, item, args.cache_dir, loader): item
            for item in index_data
        }
        for future in concurrent.futures.as_completed(future_to_item):
            res = future.result()
            results.append(res)
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"  [{completed:4d}/{total:4d}] Solves processed ({(completed/total)*100:.1f}%)")

    # Sort results by ID descending (preserving index order)
    id_order = {item["id"]: i for i, item in enumerate(index_data)}
    results.sort(key=lambda r: id_order.get(r["id"], 999999))

    # Write out dataset JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Statistics
    valid_cnt = sum(1 for r in results if r.get("is_valid"))
    unsolvable_cnt = sum(1 for r in results if r.get("validation_status") == "UNSOLVABLE")
    anomaly_cnt = sum(1 for r in results if r.get("validation_status") == "NON_ROUX_OR_ANOMALY")
    human_splits_cnt = sum(1 for r in results if r.get("human_splits"))

    print("\n" + "=" * 70)
    print("                    DATASET INGESTION SUMMARY")
    print("=" * 70)
    print(f"  Total Solves Processed:     {total}")
    print(f"  Valid Roux Solves:          {valid_cnt:4d} ({(valid_cnt/total)*100:.1f}%)")
    print(f"  Unsolvable / Parse Errors:  {unsolvable_cnt:4d} ({(unsolvable_cnt/total)*100:.1f}%)")
    print(f"  Non-Roux / Anomalies:       {anomaly_cnt:4d} ({(anomaly_cnt/total)*100:.1f}%)")
    print(f"  Human Splits Extracted:     {human_splits_cnt:4d} ({(human_splits_cnt/total)*100:.1f}%)")
    print(f"  Saved Dataset File:         {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
