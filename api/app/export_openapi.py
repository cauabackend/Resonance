"""Exporta o schema OpenAPI (usado pelo openapi-typescript no frontend): python -m app.export_openapi."""

import json
import sys

from app.main import app

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "openapi.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(app.openapi(), f, indent=2)
    print(f"OpenAPI escrito em {out}")
