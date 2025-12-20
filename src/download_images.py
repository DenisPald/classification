import csv
import hashlib
import os
import requests
from pathlib import Path

def download(csv_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("image_url", "").strip()
            if url == '':
                continue
            img_id = row["id"]

            ext = url.split(".")[-1].split("?")[0]
            filename = f"{img_id}.{ext}"
            out_path = out_dir / filename

            if out_path.exists():
                continue

            r = requests.get(url, timeout=10)
            r.raise_for_status()

            out_path.write_bytes(r.content)

if __name__ == "__main__":
    import sys
    download(Path(sys.argv[1]), Path(sys.argv[2]))
