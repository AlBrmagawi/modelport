import numpy as np

from modelport import ModelPort
from modelport.errors import ModelPortError
from modelport.validation import compare_arrays, policy_named

with ModelPort.local() as client:
    source = client.fixture()
    try:
        client.plan(source.artifact_id, device="cuda")
    except ModelPortError as error:
        assert error.code == "DEVICE_UNAVAILABLE"
        print(error.code, error.detail)
    else:
        raise AssertionError("Implicit fallback must never occur")

reference = np.array([[0.1, 0.2]], dtype=np.float32)
report = compare_arrays(reference, reference + 0.5, policy_named("fp32-default"))
assert not report["passed"]
print(report["reasons"])
