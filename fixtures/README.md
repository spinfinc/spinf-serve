# Parity fixtures

A fixture is a set of articles, each asked the same two-option cells (a template read after the article, two single-token
answers). `spinf-serve parity` scores it alone and compares every cell's two log-probabilities with a recorded reference: on the
same card architecture and the pinned dependencies the expected result is bit-identical (`lp_exact_share` 1.0), which is how a
change to a kernel, a compile boundary or a scheme is checked before it ships (`docs/BATCH_INVARIANCE.md`, `docs/PRECISION.md`).

## `wikinews_20261024.json.gz`

The public fixture: 24 Wikinews articles (250-450 words each, 8,094 words in all) x 56 cells in five template shapes
(`benchmarks/wikinews_fixture.py` builds it and lists the shapes), 1,344 cells. The file's own `lp_pos` / `lp_neg` are the
`spinf/gemma-4-12b-fp8` run (fp8s, one RTX 4090); one reference per published checkpoint sits next to it:

| file | model | scheme | recorded on |
|---|---|---|---|
| `wikinews_20261024.json.gz` | `spinf/gemma-4-12b-fp8` | fp8s | RTX 4090 |
| `wikinews_20261024.spinf-gemma-4-12b-fp8.json.gz` | the same run, as a reference file | | |
| `wikinews_20261024.spinf-qwen3.8-27b-int4.json.gz` | `spinf/qwen3.8-27b-int4` | i8 (int4-i8-split) | RTX 4090 |
| `wikinews_20261024.spinf-gemma-4-31b-fp8.json.gz` | `spinf/gemma-4-31b-fp8` | fp8s | L40S |

```sh
CUDA_VISIBLE_DEVICES=0 spinf-serve parity --fixture fixtures/wikinews_20261024.json.gz \
    --reference fixtures/wikinews_20261024.spinf-gemma-4-12b-fp8.json.gz --passes 2
SPINF_MODEL=spinf/qwen3.8-27b-int4 spinf-serve parity --fixture fixtures/wikinews_20261024.json.gz \
    --reference fixtures/wikinews_20261024.spinf-qwen3.8-27b-int4.json.gz
```

A new scheme or checkpoint records its own reference with `--record FILE` (two passes: the first pays the compile, the two must
agree). Each reference above was recorded and then verified in a fresh process: 1,344 of 1,344 cells bit-identical. The same
file is the job set of the throughput comparison (`benchmarks/compare.py --jobs fixtures/wikinews_20261024.json.gz`).

**Text licence.** The articles are Wikinews content under the Creative Commons Attribution 2.5 licence
(https://creativecommons.org/licenses/by/2.5/), written by Wikinews contributors; the fixture's `sources` list gives every
article's title, URL and the revision used. The cells (questions) are spinf's, Apache 2.0 like the rest of the repository. The
articles' `lp_pos` / `lp_neg` values are measurements, not content.

The decision model (`spinf/pplx-decider-v1-27b-int4`) reads answer codes through its readout, not free options, so it has no
parity reference on this fixture; its check is the `spinf/moderation-decider` pack's golden fixture (spinf-prompts).

The maintainers keep an older news fixture whose article text is not cleared for publication (`fixtures/news_*`, ignored by git);
every number quoted in `docs/` as "the maintainers' news fixture (18 articles x 1,328 cells)" was measured on it.
