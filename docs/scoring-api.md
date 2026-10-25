# Scoring API

## Endpoint

```
POST http://<host>:8000/v1/score
Content-Type: application/json
Authorization: Bearer <token>        # only when the server was started with --token
```

`POST /v1/chat/completions` accepts the same body, so the OpenAI SDKs work as they are: put the `scoring` block in
`extra_body`. Its response is the same, plus `"choices": []`. A chat body without a `scoring` block is refused
(`400 scoring_required`): the engine scores and extracts, it does not chat.

The hosted spinf API serves the same request and response at `https://api.spinf.com/v1/score` with an API key; the
fields below are identical there. The differences are listed at the end of this page.

## How a question works

A question is a **template**: text read after your content, ending where the answer goes, marked `{?}`. You list
the **options** (the answers) to score at that point, and the engine returns the probability of each one.

```
template: "\n\nWill {commodity} prices rise or fall next month?\nAnswer:{?}"
options:  [" rise", " fall"]
```

- `{name}` is a **placeholder**, filled from `combinations`: one template asks many questions (crude oil, natural gas,
  …). The content is read once and shared by every question.
- `{?}` must **end** the template: the model reads what comes right after it.
- Options are written as they would follow the template, usually with a **leading space** (`" rise"`). Don't put a
  space before `{?}` as well: the two spaces tokenize badly and the request is rejected (`option_boundary`).
- Write `{{` and `}}` for literal braces.

### Writing good templates

- **Content first.** The template is read after the content: start it with a line break (`"\n\n…"`) and end it right
  where the answer goes.
- **Short, distinct answers.** One word each works best (`" rise"` / `" fall"`), written as the model would continue
  the template.
- **The template is part of the question.** Rewording it changes what the model is asked, and so `p`. Naming the
  answers ("… Answer rise or fall.") lowers the residual, but it can also move the lift. Check a template on a few
  examples you know the answer to before you rely on it, and keep it fixed once you compare scores over time.
- **Skip the floor you already have.** The floor depends only on the question, so you can keep the `floor_p` of an
  earlier call and send `"empty_floor": false`. Recompute floors when `system_fingerprint` changes.

## Extraction and generation slots

A template may end with an extraction slot instead of `{?}`: the model writes the answer, decoded after the scoring
pass on the same read of the content. A query with a slot has no `options`.

| slot | the answer | ends |
|---|---|---|
| `{*}` | any text; write the template with an opening quote and name the thing: `… the company is "{*}"` | at the text after the slot (the closing quote) or a newline |
| `{n}` | a number: sign or accounting brackets, thousands separators, decimals, `thousand` / `million` / `billion`, `%` | when the next token cannot continue the number |
| `{$}` | the same with an optional currency symbol (`$ € £ ¥`), returned as `currency` | the same |

Only `{*}` may have text after it, and that text is where the answer ends: no placeholders or newlines there. An
extraction returns the most likely answers merged over spellings (text by case, punctuation and spaces; numbers by
value), each with its probability `p`, plus `mass` (the probability on all answers found) and, for numbers,
`p_negative` (the share of the top value's magnitude on its negative). Options per query: `max_tokens` (1 to 64,
default 16), `top_k` (1 to 20, default 5), `calibration: {a, b}` (adds `p_calibrated = sigmoid(a·logit(p) + b)` for the
top answer; fit `a`, `b` once per field on about 50 labelled examples). Extraction slots have no empty floor.

**Generation**: `"generate": true` on a `{*}` slot writes free text instead of searching short answers: greedy, or with
`{"temperature", "top_p", "top_k", "seed", "n"}` to sample `n` sequences reproducibly; `max_tokens` up to 100. Answers
come back as `{text, finish}` (`stop` at the template's stop text, a newline or end of text; `length` at the cap), with
no probability and no `top_k` or `calibration`.

Writing extraction templates: `{*}` in quotes, naming what goes there (`The company is "{*}"` reads as an adjective slot;
`The company in this news is "{*}"` gives the name). `{n}` / `{$}` with no quotes and no filler, ending right at the
slot. Take a number's sign from `p_negative`, not from the top answer. When the answers are a known short list, a
scored question is cheaper and has a floor.

## Request, one content

```json
{
  "model": "spinf/gemma-4-12b-fp8",
  "messages": [
    { "role": "user", "content": [ { "type": "text", "text": "OPEC+ agreed on Sunday to extend its output cuts…" } ] }
  ],
  "scoring": {
    "queries": [
      {
        "id": "direction",
        "template": "\n\nWill {commodity} prices rise or fall next month?\nAnswer:{?}",
        "options": [" rise", " fall"],
        "combinations": [ { "commodity": "crude oil" }, { "commodity": "natural gas" } ]
      },
      {
        "id": "topic",
        "template": "\n\nThe article is mainly about{?}",
        "options": [" oil", " natural gas", " coal"]
      }
    ]
  }
}
```

## Request, many contents

Send `inputs` instead of `messages`. The queries in `scoring.queries` are **shared**: every input is scored with all
of them, or only those listed in its `query_ids`. An input can add its **own** `queries` (same shape). Packing many
contents into one call is cheaper: the inputs share the forward passes, and the empty floor of a shared question is
computed once. See [Examples](examples.md#4-many-contents-in-one-call) for a full request and its response.

```json
{ "id": "a3", "messages": [ "…" ],
  "queries": [ { "id": "own", "template": "\n\nIs this good news for markets?\nAnswer:{?}", "options": [" yes", " no"] } ] }
```

## Fields

| Field | Rules |
|---|---|
| `model` | The served model's name (`GET /v1/models`): `spinf/gemma-4-12b-fp8`, `spinf/gemma-4-31b-fp8`, `spinf/qwen3.8-27b-int4`, `spinf/pplx-decider-v1-27b-int4`, an alias, or the path of the checkpoint the server was started with. Optional: a local server scores with its model when the field is absent. A server serves one model; another name → `400 model_not_served`. |
| `messages` / `inputs` | Exactly one of them. `messages` is one content (its input id is `"0"`). `inputs[].id`: unique, up to 128 characters. `inputs[].parameters`: the values of a pack's parameters for that input ([Prompt packs](prompt-packs.md#request)). |
| content | The parts of an input's messages, read as one piece of content, up to the server's content limit (default 4,096 tokens): text in order, images first, audio and video frames where you put them. A message `content` is a string or a list of parts: `{"type": "text", "text": …}`, `image_url`, `input_audio`, `video` ([Images, video and audio](#images-video-and-audio)). Roles are accepted but not used: the text is read as it is, with no chat formatting. |
| `queries[].id` | Unique within an input, up to 128 characters. |
| `queries[].template` | Up to 5 distinct placeholders `{name}` (letters, digits, `_`) and exactly one slot at the end: `{?}`, or `{*}` / `{n}` / `{$}`. Up to 256 tokens once filled (the server's query limit). |
| `queries[].options` | Scored questions: 1 to 1,000 answers, each 1 to 5 tokens, no duplicates (ignoring case when `case_insensitive`). Absent for extraction slots. |
| `queries[].combinations` | When the template has placeholders: a list of objects `{name: value}` with exactly its placeholders, or `"all"` with `values` = `{name: [value, …]}` for every combination. Up to 1,000 per query; values up to 256 characters. |
| `queries[].empty_floor` | Optional: overrides `scoring.empty_floor` for this question. |
| `queries[].group` | Optional: a named group. Per input, ungrouped queries come first, as always; each group follows in its own token trie, so its scores do not depend on the other questions of the call (prompt packs use this). |
| `queries[].max_tokens`, `top_k`, `calibration`, `generate` | Extraction slots only ([above](#extraction-and-generation-slots)). |
| `scoring.empty_floor` | Default `true`: every scored question is also scored with empty content (see below). `false`: not computed; the floor fields are `null`. |
| `scoring.case_insensitive` | Default `true`: each answer counts all its case variants (`rise`, `Rise`, `RISE`, with and without the leading space). `false`: the exact string only. |
| `scoring.packs` | Ready-made question sets scored in the same call ([Prompt packs](prompt-packs.md)). |
| `endpoint` | Accepted and echoed, for compatibility with the hosted API (where it pins a dedicated deployment); a local server ignores it. |

Default limits, all server settings ([Configuration](configuration.md)): content 4,096 tokens per input, filled template
256 tokens, 10,000 questions (templates × combinations) per input, 10 extraction slots per input, 2M processed tokens
per request, request body 2 MB.

## Images, video and audio

Content can include images, audio and video frames besides text, as OpenAI content parts, for the models that read
them (`GET /v1/models` lists each model's `media`; the server can restrict them with `--media`). They are shared by
every question of the input, like text:

```json
{ "role": "user", "content": [
  { "type": "text", "text": "The chart in this filing:" },
  { "type": "image_url", "image_url": { "url": "data:image/png;base64,iVBORw0KGgo…", "detail": "auto" } },
  { "type": "text", "text": "Listen to this clip from the earnings call." },
  { "type": "input_audio", "input_audio": { "data": "UklGRiQ…", "format": "wav" } },
  { "type": "video", "video": { "detail": "low", "frames": [ { "t": 0.0, "image_url": { "url": "data:image/jpeg;base64,…" } }, { "t": 1.0, "image_url": { "url": "…" } } ] } }
] }
```

- **Where they are read:** images are read first, before the text of the content, where the model reads them best;
  audio and video frames stay where you put them, so a short text part before them can introduce them.
- **Images:** `data:` URLs only (`png`, `jpeg`, `webp`, `gif`); the engine never fetches a remote URL. `detail` sets
  the image's token budget. Gemma models: `low` up to 70, `auto` (default) up to 280, `high` up to 1,120 tokens, plus 2
  marker tokens per image. The Qwen model reads images at its own sizes: `low` up to 256, `auto` 1,024, `high` 4,096.
- **Audio** (Gemma 4 12B only): base64 in `wav`, `mp3`, `flac`, `ogg`, `m4a` or `webm`; 1 token per 40 ms (750 per
  30 s), plus 2 marker tokens per 30 s segment; up to 600 s. Decoding needs the `ffmpeg` binary on the server.
- **Video:** frames you sample from the video, each with its time `t` in seconds, in time order, read where you put
  them as the model's own video prompt (`mm:ss <frame> mm:ss <frame> …`), each frame an image at `detail` (default
  `low`); up to 1,024 frames per video. The engine never decodes video files.
- Media count as **content tokens**, within the content limit of an input. An image is limited to 50 megapixels.

## Prompt packs

Add a ready-made, calibrated set of questions, like content moderation, with `scoring.packs`: it is scored on the same
content in the same call as your own `queries`. See [Prompt packs](prompt-packs.md).

```json
"scoring": {
  "packs": [ { "id": "spinf/moderation" } ],
  "queries": [ "… your questions …" ]
}
```

## Response

```json
{
  "id": "scr_01M3B149VXXH39YGQX5CEW3SY5",
  "object": "score",
  "created": 1790297909,
  "model": "spinf/gemma-4-12b-fp8",
  "system_fingerprint": "fp_87dd99dbf6",
  "results": [ { "input_id": "0", "queries": [ "…" ], "packs": [ "…" ] } ],
  "warnings": [],
  "usage": {
    "prompt_tokens": 112, "completion_tokens": 0, "total_tokens": 112,
    "prompt_tokens_details": { "content_tokens": 22, "text_tokens": 22, "query_tokens": 50, "floor_tokens": 40 },
    "inputs": 1, "content": { "…": "…" }, "scoring": { "…": "…" }, "processed_tokens": 112
  },
  "timing": { "wall_ms": 105, "queue_ms": 0, "compile_ms": 9, "score_ms": 96, "gpu_busy_ms": 88 }
}
```

| Field | Description |
|---|---|
| `results[]` | One entry per input, in request order; then its queries, combinations and options (or answers), in request order. |
| `options[].p` | The probability of this answer among the answers you listed (they sum to 1). |
| `options[].logprob` | The model's log probability of this answer after the filled template: the exact string, its tokens' log probabilities summed, and its case variants added when `case_insensitive`. |
| `residual` | 1 − the total probability of your answers: the mass on any other continuation. Mostly set by the template and the words; close to 1 for answers of several tokens, whose exact-string probability is small. Judge a question on `p` and the lift, and compare residuals only across contents under the same question. |
| `options[].floor_p`, `floor_residual` | The same with **empty** content: the question's built-in bias. The lift `p − floor_p` is what the content says. `null` without the empty floor. |
| extraction `answers[]`, `mass`, `p_negative`, `p_calibrated` | [Extraction and generation slots](#extraction-and-generation-slots). The query carries `kind` (`text`, `number`, `currency`) and `generate: true` when generated. |
| `warnings[]` | `low_option_mass` when a question's one-token answers take less than 10% of the probability on average (mean residual over 0.9): check the words and the template. A hint, never an error. |
| `system_fingerprint` | An opaque identifier of the engine build and configuration that produced the scores: weights revision, scheme, engine version, GPU architecture. It changes when anything that affects the scores changes. Compare it to know whether two responses are directly comparable. |
| `usage.prompt_tokens_details` | `content_tokens`: each content once (`text_tokens`: the text part of it, the rest is media). `query_tokens`: per question, the filled template plus each answer's tokens after its first. `floor_tokens`: the same for the empty floor, once per distinct question in the call. `completion_tokens`: the tokens generated by `generate` slots. |
| `usage.inputs`, `usage.content`, `usage.scoring`, `usage.extraction`, `usage.generation`, `usage.packs`, `usage.processed_tokens` | Detailed counts per kind of work: `content` (tokens, text, images and their tokens, video frames, audio seconds), `scoring` (distinct queries, combinations, options, prompts = inputs × combinations, floor prompts, query and floor tokens), `extraction` (slots by kind, answers, decode tokens), `generation` (slots, sequences, generated tokens), `packs` (count, the share of the query tokens they added), `processed_tokens` (what the processed-token limit applies to). `extraction`, `generation` and `packs` are present only when the request used them. |
| `timing` | `wall_ms`, `queue_ms` (waiting for the GPU), `compile_ms` (tokenizing, media decoding), `score_ms`, `decode_ms` (extraction), `gpu_busy_ms`. Informational: it changes with the card and the load, and is not part of the API contract. |

Response header: `X-Request-Id`.

## Errors

Every error has the same shape:

```json
{
  "error": {
    "type": "invalid_request",
    "code": "invalid_template",
    "message": "scoring.queries.0.template: {?} must end the template: text after it would never be read.",
    "param": "scoring.queries.0.template"
  },
  "request_id": "scr_01J8ZB2Q6X4M3T"
}
```

| Status | `code` | Meaning |
|---|---|---|
| 400 | `invalid_json`, `invalid_field` | Not JSON, or a field of the wrong type or shape. `param` names it. |
| 400 | `invalid_template` | The template has no slot, more than one, text after `{?}`, a bad placeholder or more than 5. |
| 400 | `unknown_placeholder`, `missing_value` | `combinations` / `values` don't match the template's placeholders. |
| 400 | `too_many_options`, `duplicate_option`, `option_too_long`, `option_boundary` | Answers: more than 1,000, repeated, over 5 tokens, or not tokenizing cleanly after the template (check the spaces). |
| 400 | `too_many_combinations`, `too_many_prompts`, `too_many_extractions` | Over 1,000 combinations in a query, over the questions per content, or over the extraction slots per content. |
| 400 | `duplicate_id`, `unknown_query_id`, `empty_queries` | Ids used twice, `query_ids` not in `scoring.queries`, an input with nothing to score. |
| 400 | `content_too_long`, `template_too_long`, `unsupported_content` | Content over the limit, a filled template over the limit, a remote image URL or an unknown part type. |
| 400 | `model_not_served`, `scoring_required` | The server serves another model; a chat body without `scoring`. |
| 400 | `media_not_supported`, `media_disabled`, `invalid_media` | The model does not read this kind of media; the server disabled it (`--media`); a part cannot be decoded. |
| 400 | `extraction_disabled`, `generation_disabled` | The server disabled extraction or generation, or the model has no LM head (a decision model). |
| 401 | `invalid_token` | The server requires a token and the request's is missing or wrong. |
| 413 | `body_too_large`, `request_too_large` | Request body over the size limit, or over the processed-token limit. |
| 429 | `concurrency_limit` | The server's queue is full. Retry after `Retry-After` seconds. |
| 500 | `engine_error` | Scoring failed. |
| 503 | `warming_up`, `model_unavailable` | The model is loading or compiling (retry after `Retry-After` seconds), or failed to load (the server log says why). |

Prompt-pack errors (`unknown_pack`, `pack_media_not_supported`, `pack_conflict`, …) are listed in
[Prompt packs](prompt-packs.md#errors).

## Other endpoints

- `GET /v1/models`: the served model: its name, `base_model`, `capabilities` (`score`, `extract`, `generate`), `media`,
  the effective limits.
- `GET /v1/packs`: the prompt packs enabled for the served model, their categories, default thresholds and decision
  modes, and the source each came from. `GET /v1/packs/{id}?version=` downloads a pack file.
- `GET /healthz` (process up), `GET /readyz` (model loaded and warm: every start-up compile done; 503 while loading or
  draining), `GET /livez` (the same, drain or not: for a local proxy's health check).
- Trusted callers only ([Configuration](configuration.md#trusted-callers)): `GET /v0/info` (what produces the scores:
  engine version, weights revision, scheme, card), `POST` / `DELETE /v0/drain` (take the server out of a load balancer's
  routing: `/readyz` answers 503 while calls in flight finish), `POST /v0/score` (the engine-internal request format).

## Differences from the hosted API

Same request, same `results`. The hosted API adds what a service needs: an API key (`Authorization: Bearer ssk-…`),
billing fields in `usage` (`billed_tokens`) and the account endpoints, rate-limit and concurrency headers, several
models behind one host, the `endpoint` field for dedicated deployments, and errors of its own (`401` key errors, `402
no_credit`, `403 model_not_enabled`, `media_not_allowed`). A client written for one runs against the other by changing
the base URL and the key; it should treat the billing fields as optional.
