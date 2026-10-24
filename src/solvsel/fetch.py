"""Download a BigSolDB release from Zenodo and record what was downloaded.

    python -m solvsel.fetch --search              # list matching Zenodo records
    python -m solvsel.fetch --record-id <id>      # download one record's files

Files land in data/raw/<record-id>/ next to MANIFEST.json, which stores the
DOI, version, license, publication date and a SHA-256 for every file. The
pipeline copies that manifest into the processed output, so every processed
dataset names the exact release it came from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ZENODO_API = "https://zenodo.org/api/records"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def search(query: str = '"BigSolDB"', size: int = 10) -> list[dict]:
    url = f"{ZENODO_API}?{urllib.parse.urlencode({'q': query, 'sort': 'mostrecent', 'size': size})}"
    hits = _get_json(url)["hits"]["hits"]
    return [
        {
            "id": h["id"],
            "title": h["metadata"].get("title"),
            "version": h["metadata"].get("version"),
            "date": h["metadata"].get("publication_date"),
            "doi": h.get("doi"),
        }
        for h in hits
    ]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(record_id: str, out_root: Path = RAW_DIR) -> Path:
    rec = _get_json(f"{ZENODO_API}/{record_id}")
    meta = rec["metadata"]
    out = out_root / str(record_id)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for f in rec["files"]:
        dest = out / f["key"]
        print(f"downloading {f['key']} ({f.get('size', '?')} bytes)", file=sys.stderr)
        urllib.request.urlretrieve(f["links"]["self"], dest)
        digest = sha256(dest)
        expected = f.get("checksum", "")
        if expected.startswith("md5:"):
            md5 = hashlib.md5(dest.read_bytes()).hexdigest()
            if md5 != expected[4:]:
                raise IOError(f"checksum mismatch for {f['key']}")
        files.append({"name": f["key"], "size": dest.stat().st_size, "sha256": digest})
    manifest = {
        "source": "zenodo",
        "record_id": str(record_id),
        "doi": rec.get("doi"),
        "title": meta.get("title"),
        "version": meta.get("version"),
        "publication_date": meta.get("publication_date"),
        "license": (meta.get("license") or {}).get("id"),
        "files": files,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out / 'MANIFEST.json'} (license: {manifest['license']})", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--search", nargs="?", const='"BigSolDB"', metavar="QUERY")
    g.add_argument("--record-id")
    p.add_argument("--out", type=Path, default=RAW_DIR)
    a = p.parse_args(argv)
    if a.search:
        for r in search(a.search):
            print(f"{r['id']}\t{r['date']}\tv{r['version']}\t{r['doi']}\t{r['title']}")
    else:
        download(a.record_id, a.out)


if __name__ == "__main__":
    main()
