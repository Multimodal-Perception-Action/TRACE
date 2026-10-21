# Environment and offline deployment

1. Create a local Python 3.10–3.12 environment with `bash install.sh`. Do not reuse a NAS virtual environment across machines with different Python installations.
2. Install graphics system libraries and NVIDIA drivers appropriate for your host. Existing CUDA driver compatibility must be verified; the package does not alter drivers.
3. Download the TRACE model and inference-resource repositories with `tools/hub_assets.py`; the downloader also obtains the official Qwen bases. The hub client is optional for fully offline deployments with local files.
4. Run `python run.py --check-models`, then the end-to-end command in the README.

The supplied dependency pins reflect the tested environment. GPU language models are loaded sequentially in separate processes to release memory between stages. The two Qwen bases occupy about 16 GB on disk; reserve additional space for caches, robot assets and outputs.

For a machine without `ensurepip`, the installer uses the included hash-checked pip wheel in `bootstrap/`. It never installs into system Python. `TRACE_VENV` changes the environment directory. A Python-version mismatch with an existing virtual environment is rejected; select a fresh directory instead.

ModelScope access to private repositories requires a read-capable token in the local client login store. Never paste tokens into public issues. If a local proxy has an unsupported `socks://` URL, correct the proxy scheme/configuration or temporarily unset proxy variables for the login command; do not disable TLS verification.

Offline inference uses local model files only and does not contact the hub. Copy the downloaded model/resource directories together with the code to the offline host. Install dependencies separately for that host; do not copy a virtual environment between machines.
