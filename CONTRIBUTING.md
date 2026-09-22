# Development

Install from the [locked CPU setup guide](docs/getting-started.md). Use `uv run --no-sync` for
Python commands and `npm ci` inside `web/`. Build the frontend once, then `modelport
serve` is a single local launcher. For frontend hot reload, run `npm run dev` in a
second terminal; Vite proxies API requests to loopback port 8765.

Run Ruff formatting/lint, mypy, the full Pytest suite, frontend type/build/unit tests,
and the real Playwright workflow. Tests use generated models and run offline after
dependencies/browser installation. Successful conversions must execute actual native
runtimes; mocks are reserved for controlled failure/resource tests.

When schemas change, run `python scripts/schema.py`, then `npm run schema` and
`npm exec prettier -- --write src/schema.d.ts` in `web/`. CI checks both generated
OpenAPI and frontend types for drift. Add an Alembic revision for schema changes;
workers must never migrate independently.

Preserve numerical policy gates when debugging. Record observed speed regressions,
unsupported operator/device conditions, and skipped hardware checks honestly.
Do not introduce a selectable adapter until its import/conversion/runtime/validation
and documentation gates are satisfied. See `docs/adapter-authoring.md`.

## Propose a change

For a larger change, open an issue describing the workflow and expected result.
Fork the repository, create a focused branch and keep the change reviewable.
Include a small reproduction or real execution evidence for runtime changes.
Update the documentation and changelog when user-visible behavior changes.

Open a pull request against `main` using the supplied template. Report checks you
actually ran, including any platform or hardware limitations. Keep discussions
respectful and focused on reproducible behavior.

## Before pushing

Run these commands from the repository root after the locked installation:

```sh
uv run --no-sync python scripts/check.py
uv run --no-sync python scripts/audit.py
cd web
npm ci
npm test
npm run build
npm exec prettier -- --check src tests scripts playwright.config.ts vite.config.ts
npm exec playwright -- install chromium
npm run e2e
npm audit --audit-level=low
cd ..
uv run --no-sync python scripts/release.py --skip-frontend
uv run --no-sync python scripts/verify_release.py
```

Use `npm.cmd` in PowerShell. Run heavy browser/native/Docker checks sequentially on
memory-constrained hosts. GitHub Actions also runs the Linux Docker suite, an
offline CPU demo, Gitleaks and an installed-wheel smoke check.

Review `git status --short` and `git diff --cached --stat` before committing. Keep
credentials, data directories, downloaded models, environments and generated
release artifacts out of the repository. `.env.example` contains configuration
examples only. Release archives belong in a separately reviewed GitHub Release.
