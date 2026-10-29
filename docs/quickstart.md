# Quickstart

## 1. Install and serve a model

You need a Linux host with an NVIDIA GPU of 24 GB or more and a driver that supports CUDA 13, Python 3.12, and about 15
GB of disk for the default model. fp8 tensor cores (Ada, Hopper or Blackwell: RTX 4090 / 5090, L4, L40S, RTX 6000 Ada,
H100) give the fastest and batch-invariant scheme; on Ampere the engine falls back to int8 weights (see
[Configuration](configuration.md)).

```bash
pip install "spinf-serve[server]"           # or: uv add "spinf-serve[server]"
spinf-serve serve spinf/gemma-4-12b-fp8     # downloads the checkpoint on first start, serves on port 8000
```

The same server as a container, with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
installed (Python and the pinned dependencies come with the image; the host needs only the driver):

```bash
docker run --gpus all -p 8000:8000 -v spinf-cache:/root/.cache ghcr.io/spinfinc/spinf-serve
docker run --gpus all -p 8000:8000 -v spinf-cache:/root/.cache ghcr.io/spinfinc/spinf-serve spinf/gemma-4-31b-fp8 --token s3cret
```

The volume at `/root/.cache` keeps the downloaded model, the compile caches and the official packs between runs, so only the
first start downloads and compiles. Arguments after the image name are those of `spinf-serve serve`; the `SPINF_*` environment
variables work too (`-e SPINF_MODEL=spinf/qwen3.8-27b-int4`). Images are tagged by version (`ghcr.io/spinfinc/spinf-serve:0.1.2`)
and `latest`.

The server answers `GET /readyz` with 200 once the model is loaded and warm: by default the warm-up runs every compile a
request could pay (about two to two and a half minutes on a 4090 for the 12B, most of it the extraction path; `--warmup
short` or `none` trades that for slower first calls, see [Configuration](configuration.md#server-options)). Add `--token
<secret>` to require `Authorization: Bearer <secret>` on every call; without it the server is open to whoever can reach
the port.

Models: `spinf/gemma-4-12b-fp8` (the default, one 24 GB card), `spinf/gemma-4-31b-fp8` (one 48 GB card),
`spinf/qwen3.8-27b-int4` (one 24 GB card). See [Configuration](configuration.md#models).

## 2. Score some content

Your content goes in `messages`, as in a chat completion. A question is a **template** that is read after the content
and ends where the answer goes (`{?}`), with the **options** (the answers) to score there. A placeholder such as
`{commodity}` asks the same question for each value you list.

```bash
curl http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spinf/gemma-4-12b-fp8",
    "messages": [ { "role": "user", "content": "OPEC+ agreed on Sunday to extend its voluntary output cuts through the first quarter, while Saudi Arabia signaled it could deepen reductions if prices keep sliding. Analysts noted that US crude inventories fell for a third straight week as refinery runs reached a seasonal high." } ],
    "scoring": {
      "queries": [
        {
          "id": "direction",
          "template": "\n\nWill {commodity} prices rise or fall next month?\nAnswer:{?}",
          "options": [" rise", " fall"],
          "combinations": [ { "commodity": "crude oil" }, { "commodity": "natural gas" } ]
        },
        {
          "id": "supply_cut",
          "template": "\n\nDoes the article report a cut in oil supply?\nAnswer:{?}",
          "options": [" yes", " no"]
        },
        {
          "id": "topic",
          "template": "\n\nThe article is mainly about{?}",
          "options": [" oil", " natural gas", " coal"]
        }
      ]
    }
  }'
```

Against the hosted API the same call is `https://api.spinf.com/v1/score` with `-H "Authorization: Bearer $SPINF_API_KEY"`.

### With the OpenAI SDK

`/v1/chat/completions` takes the same body, so the OpenAI SDKs work as they are: put the `scoring` block in
`extra_body`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")  # api_key = your --token if you set one
resp = client.chat.completions.create(
    model="spinf/gemma-4-12b-fp8",
    messages=[{"role": "user", "content": article_text}],
    extra_body={
        "scoring": {
            "queries": [
                {
                    "id": "direction",
                    "template": "\n\nWill {commodity} prices rise or fall next month?\nAnswer:{?}",
                    "options": [" rise", " fall"],
                    "combinations": [{"commodity": "crude oil"}, {"commodity": "natural gas"}],
                }
            ]
        }
    },
)
for query in resp.model_extra["results"][0]["queries"]:
    for c in query["combinations"]:
        print(query["id"], c["values"], [(o["text"], o["p"], o["floor_p"]) for o in c["options"]])
```

The scores are in `resp.model_extra["results"]` (the response's `choices` is empty: nothing is generated).

## 3. Read the scores

The response of the call above, scored by the Gemma 4 12B model (the hosted service and a local server give the same
numbers for the same engine fingerprint):

```json
{
  "id": "scr_01M3B14ABQ0S81CBK2WQ5HPNX9",
  "object": "score",
  "created": 1790297909,
  "model": "spinf/gemma-4-12b-fp8",
  "system_fingerprint": "fp_87dd99dbf6",
  "results": [
    { "input_id": "0",
      "queries": [
        { "id": "direction",
          "combinations": [
            { "values": { "commodity": "crude oil" },
              "options": [
                { "text": " rise", "logprob": -2.813903, "p": 0.654929, "floor_p": 0.473377 },
                { "text": " fall", "logprob": -3.454679, "p": 0.345071, "floor_p": 0.526623 } ],
              "residual": 0.908432, "floor_residual": 0.936882 },
            { "values": { "commodity": "natural gas" },
              "options": [
                { "text": " rise", "logprob": -2.390459, "p": 0.444475, "floor_p": 0.605271 },
                { "text": " fall", "logprob": -2.167439, "p": 0.555525, "floor_p": 0.394729 } ],
              "residual": 0.793942, "floor_residual": 0.911571 } ] },
        { "id": "supply_cut",
          "combinations": [
            { "values": {},
              "options": [
                { "text": " yes", "logprob": -0.552571, "p": 0.834359, "floor_p": 0.435885 },
                { "text": " no", "logprob": -2.169414, "p": 0.165641, "floor_p": 0.564115 } ],
              "residual": 0.310287, "floor_residual": 0.323976 } ] },
        { "id": "topic",
          "combinations": [
            { "values": {},
              "options": [
                { "text": " oil", "logprob": -2.843978, "p": 0.990817, "floor_p": 0.815715 },
                { "text": " natural gas", "logprob": -7.552238, "p": 0.008938, "floor_p": 0.064336 },
                { "text": " coal", "logprob": -11.14758, "p": 0.000245, "floor_p": 0.119949 } ],
              "residual": 0.941267, "floor_residual": 0.999882 } ] } ] }
  ],
  "warnings": [],
  "usage": { "prompt_tokens": 148, "completion_tokens": 0, "total_tokens": 148,
             "prompt_tokens_details": { "content_tokens": 50, "text_tokens": 50, "query_tokens": 49, "floor_tokens": 49 },
             "inputs": 1, "content": { "…": "…" }, "scoring": { "…": "…" }, "processed_tokens": 148 },
  "timing": { "wall_ms": 105, "…": "…" }
}
```

- `p`: the probability of each answer among the ones you listed (they sum to 1).
- `floor_p`: the same answer scored with empty content. The lift `p − floor_p` is what the content says: here the
  article makes " yes" to a supply cut go from 0.44 to 0.83, and " rise" for crude oil from 0.47 to 0.65. For natural
  gas it moves " rise" *below* its floor (0.44 against 0.61).
- `residual`: the probability that went to words other than your answers. It is mostly set by the template and the
  words, and it is naturally close to 1 for answers of several tokens (" natural gas"). Judge a question on `p` and
  the lift, and compare residuals across contents under the same question.
- `usage`: 148 tokens processed: the content once, 4 questions, their 4 empty floors; the detailed blocks and `timing` are
  in the [Scoring API](scoring-api.md#response).

See [Calibration](calibration.md) for how to use them, and [Examples](examples.md) for extraction, many contents in one
call and prompt packs.

## Tips

- **Content first, then the question.** The template is read after the content. Start it with a line break
  (`"\n\n…"`) and end it right where the answer goes.
- **Start each option with a space** (`" rise"`), and don't put a space before `{?}` as well: `"…about{?}"`, not
  `"…about {?}"` (that is refused with `option_boundary`).
- **Add a few labelled examples** to the content, before the item, when the zero-shot scores are not good enough:
  they are the biggest lever. See [Writing questions](writing-questions.md).
- **Use short, distinct answers.** One word each is best (`" rise"` / `" fall"`), and each answer should read
  naturally after the template.
- **Ask a grid in one template** with placeholders and `"combinations": "all"`.
- **Send many contents in one call** with `inputs`: the content of each is read once, and the empty floor of every
  shared question is computed once for the whole call.
- **Turn off the floor you already have.** The floor depends only on the question, so after a first call you can keep
  its `floor_p` and send `"empty_floor": false` on later calls with the same questions.
