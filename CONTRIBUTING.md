# Contributing to spinf-serve

Issues and pull requests are welcome. Before a large change, open an issue to discuss it.

- **Set up**: `uv sync --extra server`, then `uv run pytest` (the CPU suite; the official packs' golden fixtures need a
  checkout of spinf-prompts next to this one or `SPINF_PROMPTS_DIR`) and `uv run ruff check . && uv run ruff format --check .`.
- **Scores are the product.** A change that touches the engine (kernels, quantization, packing, the model stacks, the lm
  head, extraction) must keep the published parity references bit-identical on an fp8 card:
  `spinf-serve parity --fixture fixtures/wikinews_20261024.json.gz --reference fixtures/wikinews_20261024.spinf-gemma-4-12b-fp8.json.gz`.
  Say in the pull request which references you ran. A change that moves the scores is a new engine version with new
  references, decided with the maintainers.
- **The API is a contract.** The request and response are those of the hosted spinf service; a field is added, never
  changed or removed, and `docs/scoring-api.md` is updated in the same pull request.
- **Dependencies are pinned** for the engine (`torch`, `triton`, `transformers`): they change only together with a parity run.
- **Docs**: `docs/` is written for both the hosted and the self-hosted reader; keep it that way, and keep the public
  hygiene test green (`tests/test_public_hygiene.py`).
- Commit messages say what changed and why, in prose. Apache 2.0 applies to every contribution.
