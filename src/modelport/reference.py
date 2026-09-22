"""Offline API reference; no remote scripts, inline scripts or credential storage."""

import html

CSS = """
body{margin:0;background:#f4f7fb;color:#182f46;font:16px/1.6 system-ui,sans-serif}
main{max-width:1100px;margin:40px auto;padding:0 24px}a{color:#175d9d}
h1{line-height:1.2}table{width:100%;border-collapse:collapse;background:white}
td,th{text-align:left;padding:12px;border-bottom:1px solid #dce5ee;vertical-align:top}
code,pre{font:13px/1.6 ui-monospace,monospace;overflow-wrap:anywhere}
pre{background:#172e45;color:white;padding:20px;overflow:auto;border-radius:8px}
.table-wrap{overflow:auto}footer{margin:32px 0;color:#465f77}
"""


def render_reference(schema: dict) -> str:
    rows = []
    for path, methods in schema["paths"].items():
        for method, operation in methods.items():
            if method not in {"get", "post", "put", "patch", "delete", "head", "options"}:
                continue
            rows.append(
                "<tr><td><code>"
                + method.upper()
                + "</code></td><td><code>"
                + html.escape(path)
                + "</code></td><td>"
                + html.escape(operation.get("summary", ""))
                + "</td></tr>"
            )
    return (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ModelPort API reference</title><link rel="stylesheet" href="/api-reference.css">
</head><body><main><a href="/">ModelPort workspace</a><h1>API reference</h1>
<p>Version 0.1.0. The complete typed request, response and authentication contracts
are in <a href="/openapi.json">OpenAPI JSON</a>. This reference works offline.</p>
<h2>Authentication</h2><p>Send your local operator token in the
<code>Authorization: Bearer &lt;token&gt;</code> header. Retrieve it with
<code>modelport token</code> or
<code>docker compose -f docker/compose.yml exec modelport modelport token</code>.</p>
<pre>curl -H "Authorization: Bearer YOUR_OPERATOR_TOKEN" http://127.0.0.1:8765/api/v1/models</pre>
<p>Import, conversion, validation and benchmark requests return durable jobs.
Read <code>/api/v1/jobs/{job_id}</code> or stream <code>/api/v1/jobs/{job_id}/events</code>
with the same authorization header. Errors use structured problem JSON.</p>
<h2>Endpoints</h2><div class="table-wrap"><table><thead><tr><th>Method</th>
<th>Path</th><th>Operation</th></tr></thead><tbody>"""
        + "".join(rows)
        + """
</tbody></table></div><footer>Local single-operator CPU application.
Use the OpenAPI document with your preferred API client.</footer></main></body></html>"""
    )
