source .venv/bin/activate

source examples/aloha_sim/.venv/bin/activate

XLA_PYTHON_CLIENT_ALLOCATOR=platform uv run scripts/serve_policy.py --env ALOHA

CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run scripts/serve_policy.py --env ALOHA
