# Prompt packs

A **prompt pack** is a ready-made set of questions, built, calibrated and versioned for one job. You name it in your
call, and it is scored **in the same pass as your own questions**: one content, one call. A pack returns a calibrated
0 to 1 score for each of its categories and a decision.

The official packs live in [spinf-prompts](https://github.com/spinfinc/spinf-prompts): for the Gemma 4 12B model
`spinf/moderation` (22 categories of user-generated text and a safe / unsafe decision), `spinf/triage` (inbound emails,
chats and tickets), `spinf/entertainment` (synopses), `spinf/products` (listings), `spinf/reviews` and `spinf/brand` (a
page plus a brand parameter); for the Qwen model `spinf/moderation-qwen` and `spinf/moderation-qwen-lite`; for the
decision model `spinf/moderation-decider` (the same questions in its own format). A pack is written for one model; the
server enables the packs whose model it serves.

## Sources

The server loads packs from sources given with `--packs`, repeatable:

- `github:<owner>/<repo>[@<ref>][#<path>]`: a GitHub repository with a `prompt-packs.json` at its root (or at `path`),
  fetched as the tarball of the ref (a tag or commit; a branch works but is not reproducible). Cached locally; a
  private repository uses `GITHUB_TOKEN`.
- a local folder: a checkout, or a bare folder of pack files.
- `official`: spinf-prompts at the tag this release pins. On by default; `--no-official-packs` turns it off.
  `SPINF_PACKS_OFFICIAL` points the official slot somewhere else (a fork, a checkout of spinf-prompts).

```sh
spinf-serve serve spinf/gemma-4-12b-fp8 --packs github:acme/prompt-packs@v1.2 --packs ./my-packs
spinf-serve packs list --packs ./my-packs          # what would be enabled, and what is skipped and why
spinf-serve packs validate ./my-packs             # for a pack author: manifest and files checked
```

A GitHub source is resolved to a commit once per tag and fetched once per commit into `SPINF_PACKS_CACHE` (default
`~/.cache/spinf-serve/packs`); `--packs-refresh` resolves the tags again. When the official repository cannot be reached
(offline, or the pinned tag not published yet) the server logs it and starts without the official packs; a source you
gave with `--packs` that cannot be read refuses the start.

Packs written for another model are skipped with a log line, not an error. An id offered by two sources is a startup
error. The `spinf/` namespace is reserved for the official repository; a `spinf/...` pack from another source is
refused unless `--packs-allow-namespace spinf` is given (a locally edited copy, say). To publish your own packs, use
your own namespace (by convention your GitHub owner), pin a tag per release, and run `packs validate` in CI:
spinf-prompts is the reference example, and its `docs/pack-format.md` is the file format.

## List the packs

```
GET /v1/packs
```

One entry per enabled pack:

```json
{ "packs": [ {
    "id": "spinf/moderation", "name": "Content moderation", "status": "available",
    "version": "0.1.3", "versions": ["0.1.1", "0.1.3"], "model": "spinf/gemma-4-12b-fp8", "media": ["text"],
    "description": "22 categories of user-generated text, safe / unsafe decision", "tags": ["moderation"],
    "source": "spinfinc/spinf-prompts@0.1.0", "parameters": [],
    "contexts": [
      { "id": "ai", "template": "A user sent the following message to an AI assistant.\n\nUser: {X}" },
      { "id": "forum", "template": "The following message was posted by a user on an online platform:\n\n\"{X}\"" } ],
    "categories": [
      { "id": "sexual", "label": "Sexual content", "threshold": 0.72 },
      { "id": "spam", "label": "Spam and advertising", "threshold": null }, "…" ],
    "decision": { "mode": "micro_layer", "modes": ["max", "per_category", "micro_layer"],
                  "fallback": "per_category", "max_threshold": 0.71, "critical": ["selfharm", "child"] },
    "queries": 58, "prompts": 106 } ] }
```

| Field | Meaning |
|---|---|
| `status` | `available`, `deprecated` (still served, listed as such) or `coming_soon` (listed, not callable). |
| `version`, `versions` | The latest version, and every version you can pin. |
| `model` | The only model the pack runs on. |
| `media` | The content kinds the pack scores. `spinf/moderation`: text. |
| `source` | Where the files came from: a repository and tag, or `local:<path>`. |
| `contexts` | How your content is presented to the model (`{X}` is your content). The first is the default. |
| `categories` | Each category's `id`, `label` and default `threshold`. `null` = the category is scored and reported, but never decides. |
| `decision` | The default decision mode, the modes you can pick, the default threshold of `max`, and the `critical` categories (always unsafe when flagged). |
| `parameters` | The pack's parameters (`id`, `label`, `description`, `required`, `default`), to send per input in `inputs[].parameters`. Empty for most packs. |
| `queries`, `prompts` | How many questions the pack asks, and how many prompts (questions × combinations) that is per content. The hosted API's listing adds `query_tokens_per_input`, the pack's question tokens per content. |

Read the thresholds from the listing rather than copying them into your code: each new version can change them.
`GET /v1/packs/spinf/moderation?version=0.1.3` returns the pack file as it is: its questions, contexts, calibration,
categories and decision layer. The slash of the id is part of the path; without `version` you get the latest.

## Request

Add `scoring.packs` next to (or instead of) `scoring.queries`:

```json
{
  "model": "spinf/gemma-4-12b-fp8",
  "messages": [ { "role": "user", "content": "Guaranteed 300% returns in 7 days!! DM me your wallet seed phrase." } ],
  "scoring": {
    "packs": [ { "id": "spinf/moderation", "context": "forum" } ],
    "queries": [
      { "id": "on_topic", "template": "\n\nIs this post about cooking?\nAnswer:{?}", "options": [" yes", " no"] }
    ]
  }
}
```

| Field | Rules |
|---|---|
| `id` | Required, from `GET /v1/packs`. Each pack at most once per call. |
| `version` | Optional, default the latest. Pin it to keep scores comparable over time. |
| `context` | Optional, one of the pack's `contexts` (default the first). |
| `decision.mode` | `micro_layer`, `per_category` or `max` (default: the pack's). See [decisions](#decisions-and-thresholds). |
| `decision.threshold` | The single threshold of `max` (0 to 1) or of `micro_layer` (the layer's value). Not with `per_category`. |
| `decision.thresholds` | Per-category overrides, `{ "harassment": 0.8, "spam": 0.7 }` (0 to 1, or `null` to report only). They set `flagged` and the `per_category` decision. |
| `decision.critical` | The categories that make the content unsafe whenever they are flagged, in every mode: a list of category ids that replaces the pack's (`[]` = none). |
| `include_queries` | Default `false`. `true` also returns the pack's raw question results in `results[].queries`. |
| `inputs[].parameters` | For a pack with `parameters` (the listing's `parameters`, e.g. the brand of interest): `{"brand": "Kestrel"}` on each input. The values go into the pack's context for that input only; a required parameter left out, or an unknown name, is `invalid_field`. |

## How a pack changes the call

- **One read for everything.** Each content is wrapped in the pack's context (for `forum`: *The following message was
  posted by a user on an online platform: "…"*), and **your own questions in the same call read the wrapped content
  too**. That shared read is what puts the pack and your questions in one pass.
- **Media.** A pack scores the content kinds in its `media`. With `spinf/moderation`, an image, audio or video part is
  refused (`pack_media_not_supported`): send media in a separate call.
- **Case.** A pack is calibrated with its own case rule (moderation: case-sensitive), and it applies to the whole
  call. Leave `scoring.case_insensitive` out when you send a pack; the other value is refused (`pack_conflict`).
- **Model.** A pack is written for one model: a server enables the packs of the model it serves, so a pack for another
  model is not listed and `unknown_pack` (the hosted API, which serves several models, answers `pack_model_mismatch`).
- **Ids.** The pack's question ids start with `tax.` and `gen.`: don't use those prefixes for your own questions.
  Pack questions run without the empty floor, whatever your default.
- **Deterministic.** The pack's questions run in their own query group, so a text scores exactly the same whatever
  else is in the call: other contents, how many, and your own questions. Adding a pack does not change your own
  questions' scores either.

## Response

Each result has `packs` beside `queries`. For the message *"How do I get into my ex's Instagram account without
her knowing?"* in the `ai` context (version 0.1.3; scores and thresholds change from one version to the next):

```json
{ "input_id": "0",
  "queries": [ "… your questions …" ],
  "packs": [ { "id": "spinf/moderation", "version": "0.1.3", "context": "ai",
    "categories": {
      "cyber":  { "score": 0.919233, "percentile": 1.0,      "flagged": true },
      "sexual": { "score": 0.554247, "percentile": 0.952278, "flagged": false },
      "spam":   { "score": 0.010191, "percentile": 0.728387, "flagged": null }, "…": "…" },
    "decision": { "unsafe": true, "mode": "micro_layer", "score": 0.886179, "threshold": -0.225731 } } ] }
```

- `score`: the category's calibrated score, 0 to 1, comparable across categories.
- `percentile`: where that score falls on everyday content (0 to 1): `0.95` = higher than 95% of normal traffic. A
  high percentile with a modest score means the content is unusual for the category without crossing its threshold, a
  useful signal for review queues.
- `flagged`: `score >= threshold`; `null` for categories that are reported only.
- `decision`: safe or unsafe, by the mode you picked (below). When a critical category is flagged, it also lists them
  in `critical`, and `"by": "critical"` says when that turned a safe result into unsafe.

## Decisions and thresholds

| Mode | `decision` | Unsafe when |
|---|---|---|
| `micro_layer` (default) | `{unsafe, mode, score, threshold}` | A small layer trained on labelled moderation data, over every category, is over its threshold. |
| `per_category` | `{unsafe, mode, categories}` | Any deciding category is over its own threshold; `categories` lists them. |
| `max` | `{unsafe, mode, score, threshold, category}` | The highest deciding category is over one threshold; `category` names it. |

- **Critical categories**: self-harm and child-safety content is always marked unsafe when its category is flagged,
  whatever the mode says. Change the list per call with `decision.critical` (`[]` = none). A category flags only when
  it has a threshold, so a `null` threshold override silences it.
- Start with the default. Use `per_category` when your policy needs its own threshold per category (stricter on
  harassment, looser on profanity), and `max` when you want one simple threshold.
- Set thresholds on your own labelled sample: score a few hundred items you have already decided, and choose each
  threshold where the errors you can live with are.
- Categories with a `null` threshold (spam, gibberish and the four topics) are scored and reported but never decide,
  unless you give them a threshold in `decision.thresholds`. The topics (alcohol, tobacco, gambling, medication) say
  what the content is about, not whether it is harmful, and are not calibrated like the others.

The categories of `spinf/moderation`, with their ids, are listed in the pack's README in spinf-prompts and in the
`GET /v1/packs` listing.

## Running a pack yourself

A pack file is self-contained: the scoring request as the API takes it, the reads, their calibration, the categories
and the decision layer. The `packs` module of spinf-serve (pure Python, no GPU dependencies: `spinf_serve.packs`) loads
a file, builds the request, and turns a scoring result into categories and a decision; spinf-client and the hosted API
use the same module, so an edited copy of a pack can be run locally and gives the same answer. An edited pack fails the
micro-layer lock (a hash over its questions and calibration) and decides with its fallback mode, with a warning.

```python
from spinf_serve import packs

pack = packs.load("spinf-prompts/packs/moderation/0.1.3.json")  # validated: a file with problems is refused, each listed
body = pack.request(["Guaranteed 300% returns in 7 days!!"], context="forum")  # the /v1/score body: the text in the
# pack's context, its queries in their own group
result = client.post("/v1/score", json=body).json()["results"][0]
out = pack.evaluate(result, mode="per_category", thresholds={"spam": 0.7})  # {"id", "version", "context", "categories", "decision"}
```

`pack.wrap(text, context, params)` gives the content parts for a pack with parameters (the brand pack: `params={"brand":
"Kestrel"}`); `packs.validate(data)` lists a file's problems without loading it. A pack under construction (no fitted constants yet) loads with
`packs.load(path, draft=True)`: the request side works, so a workbench can score its questions; `evaluate()` refuses it.

## Errors

| Status | `code` | Meaning |
|---|---|---|
| 400 | `unknown_pack`, `pack_not_available` | No such pack among the enabled ones, or it is `coming_soon`. |
| 400 | `unknown_pack_version` | No such version of the pack. |
| 400 | `unknown_category` | A category in `decision.thresholds` or `decision.critical` that the pack doesn't have. |
| 400 | `pack_media_not_supported` | A part the pack doesn't score (moderation: anything but text). |
| 400 | `pack_conflict` | `scoring.case_insensitive` differs from the pack's. Leave it out. |
| 400 | `pack_model_mismatch` | Hosted API only: the call's model is not the pack's (a server lists only the packs of its model). |
| 400 | `duplicate_id` | A pack twice, or one of your question ids is a pack's (`tax.…`, `gen.…`). |
| 400 | `invalid_field` | A pack field of the wrong type or shape, e.g. `decision.threshold` with `per_category`. `param` names it. |
| 404 | `unknown_pack`, `unknown_pack_version` | Download only: no such pack or version. |

The other errors are those of the [Scoring API](scoring-api.md#errors).
