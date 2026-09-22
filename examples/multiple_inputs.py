import numpy as np

from modelport import ModelPort
from modelport.errors import ModelPortError

with ModelPort.local() as client:
    source = client.fixture("dual-input")
    converted = client.convert(client.plan(source.artifact_id))
    with client.load(converted.artifact_id, require_validated=True) as model:
        for batch in (2, 7):
            arrays = {name: np.zeros((batch, 64), np.float32) for name in ("left", "right")}
            outputs = model.predict(arrays).outputs
            assert outputs["logits"].shape == (batch, 10)
            assert outputs["features"].shape == (batch, 128)
        try:
            model.predict(
                {"left": np.zeros((2, 64), np.float32), "right": np.zeros((3, 64), np.float32)}
            )
        except ModelPortError as error:
            assert error.code == "INVALID_SHAPE"
            print(error.code, error.detail)
        else:
            raise AssertionError("Inconsistent batch shapes must be rejected")
