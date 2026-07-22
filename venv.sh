source .venv/bin/activate

source examples/aloha_sim/.venv/bin/activate

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py --env ALOHA
