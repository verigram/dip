


test:
    uv run pytest

# --- Linting & Formatting with Ruff ---
# Run static checks (no changes)
ruff-check:
    uv run ruff check
    uv run ruff format --check

# Apply safe auto-fixes
ruff-fix:
    uv run ruff check --fix
    uv run ruff format
