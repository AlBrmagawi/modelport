from typing import Any


class ModelPortError(Exception):
    def __init__(self, code: str, detail: str, *, status: int = 422, **details: Any):
        self.code, self.detail, self.status, self.details = code, detail, status, details
        super().__init__(detail)

    def problem(self, request_id: str = "", job_id: str | None = None) -> dict[str, Any]:
        return {
            "type": f"urn:modelport:error:{self.code.lower().replace('_', '-')}",
            "title": self.code.replace("_", " ").capitalize(),
            "status": self.status,
            "code": self.code,
            "detail": self.detail,
            "request_id": request_id,
            "job_id": job_id,
            "retryable": self.code in {"INTERRUPTED", "TIMEOUT", "RESOURCE_EXHAUSTED"},
            "details": self.details,
        }
