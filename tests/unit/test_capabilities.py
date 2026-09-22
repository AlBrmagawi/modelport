from modelport.doctor import current_fingerprint, dependency_issues
from modelport.domain import CapabilitySnapshot


def test_dependency_constraints_reject_wrong_cpu_build_and_unsupported_versions():
    versions = {
        "torch": "2.13.0+cpu",
        "onnx": "1.23.0",
        "onnxruntime": "1.30.0",
        "onnxscript": "0.7.2",
    }
    assert dependency_issues(versions) == []
    for value in ("2.10.0+cpu", "2.13.0+cu130", "not-a-version", None):
        assert dependency_issues({**versions, "torch": value})[0]["package"] == "torch"
    assert dependency_issues({**versions, "onnxruntime": "2.0.0"})[0]["package"] == "onnxruntime"


def test_fingerprint_survives_browser_integral_number_roundtrip():
    snapshot = CapabilitySnapshot(
        fingerprint="",
        checked_at="2026-09-22",
        packages={},
        environment={"ram_bytes": 1024.0},
        providers=[],
        adapters=[],
        probes={"metrics": [0.0, -0.0, 1.0, 1e-7, 0.25]},
    )
    browser = snapshot.model_copy(deep=True)
    browser.environment["ram_bytes"] = 1024
    browser.probes["metrics"] = [0, 0, 1, 1e-7, 0.25]
    assert current_fingerprint(browser) == current_fingerprint(snapshot)
    browser.probes["metrics"][-1] = 0.5
    assert current_fingerprint(browser) != current_fingerprint(snapshot)
