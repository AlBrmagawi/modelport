"""Disposable real API + worker for browser tests. Never serves production data."""

import os
import tempfile
from pathlib import Path

import numpy as np
import uvicorn

from modelport.api import create_app
from modelport.sdk import ModelPort
from modelport.security import save_npz, write_json

root = Path(__file__).resolve().parents[1]
os.environ["MODELPORT_TOKEN"] = "browser-test-only-operator-credential"
os.environ["MODELPORT_ALLOWED_ORIGINS"] = "http://127.0.0.1:8766"
with tempfile.TemporaryDirectory(prefix="modelport-browser-") as temporary:
    directory = Path(temporary)
    with ModelPort.local(directory / "fixture-data") as fixture_client:
        source = fixture_client.fixture()
        fixture = fixture_client.store.path(source.artifact_id)
        (root / ".tools").mkdir(exist_ok=True)
        calibration, validation = directory / "calibration.npz", directory / "validation.npz"
        save_npz(
            calibration,
            {
                "images": np.random.default_rng(505)
                .standard_normal((512, 1, 8, 8))
                .astype(np.float32)
            },
        )
        save_npz(
            validation,
            {
                "images": np.random.default_rng(9999)
                .standard_normal((4, 1, 8, 8))
                .astype(np.float32)
            },
        )
        write_json(
            root / ".tools" / "e2e-info.json",
            {
                "fixture": str(fixture),
                "calibration": str(calibration),
                "validation": str(validation),
            },
        )
    app = create_app(directory / "api-data")
    uvicorn.run(app, host="127.0.0.1", port=8766, log_level="warning")
