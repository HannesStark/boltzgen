### Release process

- Update the version in `pyproject.toml` (e.g., `0.2.0`).
- Create a GitHub release:
  - Tag the release as `vX.Y.Z` (e.g., `v0.2.0`) matching the version in `pyproject.toml`.
  - Note: we don’t auto-check this; keep them in sync.
- Publishing the GitHub release will automatically publish to PyPI via a GitHub Action.

