"""Opt-in live public downloads. Does not redistribute model weights or use tokens."""

import argparse
from pathlib import Path

import numpy as np

from modelport.domain import HubImport
from modelport.remote import hub_request
from modelport.sdk import ModelPort
from modelport.security import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = []
    with ModelPort.local(args.data_dir) as client:
        for filename in ("hub-mnist.json", "hub-tiny-bert.json"):
            manifest = read_json(root / "examples" / filename)
            hub = HubImport.model_validate(manifest)
            artifact = client.import_hub(hub.repository, hub.revision, manifest["files"])
            inputs = (
                {artifact.descriptor.inputs[0].name: np.zeros((1, 1, 28, 28), np.float32)}
                if "mnist" in filename
                else {
                    "input_ids": np.array([[2, 5, 10, 3], [2, 8, 20, 3]], np.int64),
                    "attention_mask": np.ones((2, 4), np.int64),
                    "token_type_ids": np.zeros((2, 4), np.int64),
                }
            )
            with client.load(artifact.artifact_id) as model:
                outputs = model.predict(inputs).outputs
                assert all(np.isfinite(array).all() for array in outputs.values())
            https = client.import_https(hub_request(hub).model_dump()["files"])
            assert https.artifact_id == artifact.artifact_id
            evidence.append(
                {
                    "example": filename,
                    "artifact_id": artifact.artifact_id,
                    "inputs": {name: list(value.shape) for name, value in inputs.items()},
                    "outputs": {name: list(value.shape) for name, value in outputs.items()},
                    "https_identity_matches_hub": True,
                    "origin": artifact.manifest.get("origin"),
                }
            )
    write_json(args.data_dir / "remote-verification.json", evidence)
    print("Verified public HTTPS and pinned Hub imports; finite MNIST and BERT CPU predictions.")


if __name__ == "__main__":
    main()
