# Reproducible authoring environment

Upstream: https://github.com/piskvorky/smart_open
Upstream commit: 03a34a0e6176f994b12f9478aba0cc6810de3e29
Verified runtime: 3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]
Verified platform: win32

Use an isolated virtual environment. Install the pinned dependencies with
`python -m pip install -r requirements-domain.lock`, then `python -m pip install -e .`.
Set the following environment variables before installation and testing:

```text
PYTHONUTF8=1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

Core baseline:

```text
python -m pytest -q -o addopts= tests/test_http.py tests/test_bytebuffer.py
python domain_baseline.py
```

Measured pytest process time: 3.145 seconds on the authoring host.
Only the documented core test selection is claimed; optional integrations are outside this environment.

All documented checks use in-memory fixtures or local temporary files. No production
accounts, data services, credentials or external network endpoints are required after setup.
Reset by making a fresh checkout of this seed and creating a new virtual environment.
Test fixtures remain under their upstream repository license.
