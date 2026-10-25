# Benchmarks

`compare.py` sends the same questions to spinf-serve and to an OpenAI-compatible completions server (vLLM with prefix
caching: the usual alternative) over HTTP, and times both. A job is one content with N questions, each a template read
after the content with single-token answers; the completions server gets one `/v1/completions` request per question
(`max_tokens` 1, `logprobs` 20) with many in flight, so its prefix cache reuses the content; spinf-serve gets the jobs in
one request and answers them in packed passes. The warm pass of each is the number (the first passes pay compilation and
cache fills).

The job set is the public parity fixture, `fixtures/wikinews_20261024.json.gz` (24 Wikinews articles x 56 two-option
questions = 1,344 questions; `fixtures/README.md`), which `compare.py` reads directly; `wikinews_fixture.py` is the script
that built it. A JSONL file works too: one line per content, `{"id", "text", "questions": [{"template": "\n\n...{?}",
"options": [" a", " b"]}]}`; the template may hold `{content}` where the content goes (without it the content comes first),
and the options must be single tokens for the completions side (it reads them from the top-20 next-token list).

```sh
# vLLM 0.26, Gemma 4 12B quantized to fp8 on load, prefix caching on (the default in vLLM v1), one 24 GB card
docker run -d --name vllm --gpus '"device=0"' --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -p 8033:8000 \
  vllm/vllm-openai:v0.26.0 --model google/gemma-4-12B --quantization fp8 --max-model-len 2048 --max-num-seqs 128 \
  --gpu-memory-utilization 0.92 --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"audio":0}'
python benchmarks/compare.py --jobs fixtures/wikinews_20261024.json.gz --vllm http://localhost:8033 --model google/gemma-4-12B --workers 128

# spinf-serve, same card, after stopping vLLM. The jobs go through /v0/score (a trusted caller's endpoint) in one body of
# ~3.4 MB, so raise the body cap for the run
spinf-serve serve spinf/gemma-4-12b-fp8 --port 8040 --trusted-token bench --max-body-bytes 64000000
python benchmarks/compare.py --jobs fixtures/wikinews_20261024.json.gz --spinf http://localhost:8040 --spinf-token bench --passes 3
```

## Results, 2026-10-24 (the public job set)

One RTX 4090 (24 GB), driver with CUDA 13, both servers alone on the card. Jobs: 24 Wikinews articles (mean 467 tokens),
56 questions each (1,344 in all), two one-token answers per question; 30,770 tokens of work (content once + the
questions' own tokens), 647,320 tokens if the content were re-read per question (21x).

| server | warm pass | questions/s | tokens of work/s | re-read equivalent tok/s |
|---|---|---|---|---|
| spinf-serve 0.1, Gemma 4 12B fp8 static scales, `/v0/score` | **2.74 s** | **491** | **11,242** | **236,495** |
| vLLM 0.26, Gemma 4 12B `--quantization fp8`, prefix caching, 128 in flight | 6.72 s | 200 | 4,579 | 96,336 |
| vLLM 0.26, same, 64 in flight | 7.34 s | 183 | 4,192 | 88,188 |

spinf-serve is 2.5x faster than vLLM at vLLM's best setting (its `--max-num-seqs`). vLLM's prefix cache was warm from its
first pass (7.57 s). spinf-serve's first two passes took 37.6 and 53.6 s: at the time the server reported ready before its
extraction warm-up had finished, so both passes waited behind that compile (fixed the same day: `/readyz` now waits for the
whole warm-up, and a first pass after it runs at the warm speed). The ratio depends on the questions per content: the more
questions share one read, the larger the gap.

### Earlier run, 2026-10-12 (the maintainers' news fixture)

Same card and servers on 18 news articles (mean 308 tokens) x 74 questions (1,328 in all; 21,080 tokens of work, 449,747
re-read): spinf-serve 1.89 s / 704 questions/s / 11,179 tok/s of work; vLLM at 128 in flight 5.42 s / 245 / 3,886 (64 in
flight 6.26 s, 32 in flight 7.01 s): 2.9x. That fixture's article text is not published. Internal measurements on 2,211
decision-benchmark jobs with the int4 Qwen3.8 27B model gave 2.2x to 2.8x against vLLM 0.26 (GPTQ Marlin, prefix caching,
8 in flight; `docs/QWEN.md` §4).
