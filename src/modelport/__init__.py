"""ModelPort public SDK; ML frameworks are imported only inside workers."""

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "ModelPort":
        from modelport.sdk import ModelPort

        return ModelPort
    raise AttributeError(name)
