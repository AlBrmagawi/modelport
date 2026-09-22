"""Run from a clean wheel installation outside the checkout, using disposable data."""

import argparse
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import modelport
from modelport.api import create_app
from modelport.sdk import ModelPort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    assert "site-packages" in str(modelport.__file__)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.work_dir) as directory:
        app = create_app(Path(directory) / "data", start_worker=False)
        with TestClient(app) as http:
            response = http.get("/")
            assert response.status_code == 200 and "ModelPort" in response.text
            assert "MIT" in http.get("/third-party-licenses.txt").text
            assert "Authorization: Bearer" in http.get("/docs").text
        with ModelPort.local(Path(directory) / "data") as client:
            snapshot = client.doctor()
            verified = [adapter["id"] for adapter in snapshot.adapters if adapter["probe_verified"]]
            assert len(verified) == 5, verified
            print(
                "Installed wheel serves bundled dashboard and licenses; all five CPU probes pass."
            )


if __name__ == "__main__":
    main()
