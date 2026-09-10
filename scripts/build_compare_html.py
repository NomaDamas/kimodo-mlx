#!/usr/bin/env python3
from pathlib import Path
import json
ROOT=Path("evidence/compare")
OUT=Path("docs/compare.html")
TPL=Path("docs/compare.template.html")

def load(name):
    path=ROOT/name
    return json.loads(path.read_text()) if path.is_file() else None

def main():
    timings=load("timings.json") or {"backends":[],"quality":{},"prompt":"walk forward","frames":30,"steps":10,"seed":42}
    clips={k:load(f"{k}.json") for k in ("mlx-metal","cpp-metal","cpp-cpu")}
    payload={"timings":timings,"clips":clips}
    html=TPL.read_text().replace("__PAYLOAD__", json.dumps(payload))
    OUT.write_text(html)
    print("wrote", OUT, OUT.stat().st_size)

if __name__=="__main__":
    main()
