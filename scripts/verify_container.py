"""Verify the running local Compose service and its persistence across one restart."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = ["docker", "compose", "-f", "docker/compose.yml"]
    env = {**os.environ, "MODELPORT_PORT": str(args.port)}
    credential = subprocess.run(
        [*command, "exec", "-T", "modelport", "modelport", "token"],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    token = json.loads(credential.stdout)["token"]
    with httpx.Client(
        base_url=f"http://127.0.0.1:{args.port}",
        headers={"Authorization": "Bearer " + token},
        timeout=10,
    ) as client:
        assert client.get("/health/ready").status_code == 200
        before = client.get("/api/v1/models").json()
        assert len(before) >= 4, "Run the CPU demo in this workspace first"
        assert client.get("/").status_code == 200
        assert "MIT" in client.get("/third-party-licenses.txt").text
        assert "Authorization: Bearer" in client.get("/docs").text
        subprocess.run(
            [*command, "restart", "modelport"], cwd=root, env=env, check=True, timeout=60
        )
        deadline = time.monotonic() + 90
        while True:
            try:
                if client.get("/health/ready").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("Restarted container did not become ready")
            time.sleep(0.5)
        after = client.get("/api/v1/models").json()
        assert {item["artifact_id"] for item in before} == {item["artifact_id"] for item in after}
        caps = client.get("/api/v1/capabilities")
        assert caps.status_code == 200
        assert sum(bool(item["probe_verified"]) for item in caps.json()["adapters"]) == 5
        evidence = {
            "ready": True,
            "artifact_count": len(after),
            "restart_preserved_artifacts": True,
            "cpu_probes": 5,
            "dashboard_and_notices": True,
            "port": args.port,
        }
        target = root / ".tools" / "docker-evidence" / "service-verification.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evidence))


if __name__ == "__main__":
    main()
