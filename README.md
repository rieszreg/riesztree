# riesztree — moved into the rieszreg monorepo

> **This repository has been archived (read-only).** Active development of `riesztree` continues in the consolidated rieszreg monorepo at:
>
> **https://github.com/rieszreg/rieszreg/tree/main/packages/riesztree**

## Why

The six sibling repos (`rieszreg`, `rieszboost`, `krrr`, `forestriesz`, `riesznet`, `riesztree`) were consolidated into a single `uv`-workspace monorepo to eliminate the cross-package coordination overhead that had grown around shared abstractions in `rieszreg`. The pattern follows LangChain's monorepo at github.com/langchain-ai/langchain.

Each package is still published independently to PyPI (`pip install riesztree` continues to work). Only the source location changed.

## What you should do

- **Cloning for development**: `git clone https://github.com/rieszreg/rieszreg.git` instead of this repo. Then `uv sync --all-packages --all-extras` from the repo root.
- **Filing issues / opening PRs**: use https://github.com/rieszreg/rieszreg/issues — this archived repo no longer accepts new issues.
- **Pinning a specific version**: tags for this package now use the `riesztree-vX.Y.Z` prefix in the new monorepo (e.g. `riesztree-v0.1.0`). Old tags in this archived repo (e.g. `v0.1.0`) remain valid as pre-migration historical pins.

The full pre-migration history of this package was preserved into the monorepo via `git filter-repo`. You can browse it under `packages/riesztree/` in the new repo, with `git log --follow` working through the move.
