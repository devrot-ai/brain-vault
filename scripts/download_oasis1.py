#!/usr/bin/env python
"""Download the official OASIS-1 cross-sectional MRI release (resumable).

Fetches the 12 raw imaging discs (~15.5 GB total) and the official 2024
metadata spreadsheet from the Washington University hosts linked on the
OASIS-1 page (https://sites.wustl.edu/oasisbrains/home/oasis-1/).

Downloads are per-disc resumable (HTTP Range into ``<name>.part``), verified
against the server's Content-Length, and sequential to be gentle on the
public server. Safe to re-run after interruption: finished discs are skipped.

Usage:
    python scripts/download_oasis1.py [--out data/raw/oasis1] [--discs 1-12]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://download.nrg.wustl.edu/data/"
METADATA_URL = ("https://sites.wustl.edu/oasisbrains/files/2024/04/"
                "oasis_cross-sectional-5708aa0a98d82080.xlsx")
DISCS = list(range(1, 13))


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length", 0)) or None
    except Exception:
        return None


def download(url: str, dest: Path, log_every: float = 20.0) -> None:
    """Download ``url`` to ``dest``, resuming from ``dest.with_suffix('.part')``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    total = _remote_size(url)
    if dest.exists() and total and dest.stat().st_size == total:
        print(f"  [skip] {dest.name} complete ({total / 1e9:.2f} GB)", flush=True)
        return
    resume_from = part.stat().st_size if part.exists() else 0
    if total and resume_from >= total:
        part.rename(dest)
        print(f"  [done] {dest.name} (resumed to completion)", flush=True)
        return

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")
    t0 = time.time()
    last_log = 0.0
    mode = "ab" if resume_from else "wb"
    with urllib.request.urlopen(req, timeout=120) as resp, open(part, mode) as f:
        got = resume_from
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            now = time.time()
            if now - last_log >= log_every:
                pct = f" {got / total * 100:5.1f}%" if total else ""
                rate = (got - resume_from) / max(now - t0, 1e-6) / 1e6
                print(f"  {dest.name}{pct} {got / 1e9:5.2f} GB "
                      f"({rate:.1f} MB/s)", flush=True)
                last_log = now
    if total and part.stat().st_size != total:
        raise RuntimeError(f"{dest.name}: size mismatch after download "
                           f"({part.stat().st_size} != {total})")
    part.rename(dest)
    print(f"  [done] {dest.name} ({dest.stat().st_size / 1e9:.2f} GB "
          f"in {time.time() - t0:.0f}s)", flush=True)


def parse_discs(spec: str) -> list[int]:
    m = re.fullmatch(r"(\d+)-(\d+)", spec)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(x) for x in spec.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/oasis1")
    ap.add_argument("--discs", default="1-12", help="e.g. 1-12 or 1,3,5")
    ap.add_argument("--no-metadata", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.no_metadata:
        print("metadata:", flush=True)
        try:
            download(METADATA_URL, out / "oasis_cross-sectional.xlsx")
        except Exception as exc:  # noqa: BLE001 - discs must proceed
            print(f"  [warn] metadata: {exc}", flush=True)

    for disc in parse_discs(args.discs):
        url = f"{BASE}oasis_cross-sectional_disc{disc}.tar.gz"
        print(f"disc {disc}:", flush=True)
        try:
            download(url, out / f"oasis_cross-sectional_disc{disc}.tar.gz")
        except Exception as exc:  # noqa: BLE001 - keep downloading other discs
            print(f"  [error] disc {disc}: {exc}", flush=True)
    print("download run finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
