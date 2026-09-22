"""Generate or check the stable OpenAPI contract without loading ML frameworks."""

import argparse
import json
import tempfile
from pathlib import Path

from modelport.api import create_app

parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
with tempfile.TemporaryDirectory() as directory:
    app = create_app(directory, start_worker=False)
    schema = json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"
    app.state.client.close()
path = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"
if args.check:
    if path.read_text(encoding="utf-8") != schema:
        raise SystemExit("OpenAPI drift: regenerate scripts/schema.py and npm run schema")
else:
    path.write_text(schema, encoding="utf-8")
