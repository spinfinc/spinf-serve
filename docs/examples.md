# Examples

Every example is a `curl` call to a local server (`spinf-serve serve spinf/gemma-4-12b-fp8`). Against the hosted API,
replace the host with `https://api.spinf.com` and add `-H "Authorization: Bearer $SPINF_API_KEY"`; the body does not
change. The responses with numbers are real responses of the Gemma 4 12B model; the ones marked *shape* show the fields
with illustrative values.

## 1. One question

The smallest call: one content, one question, two answers.

```bash
curl http://localhost:8000/v1/score -H "Content-Type: application/json" -d '{
  "model": "spinf/gemma-4-12b-fp8",
  "messages": [ { "role": "user", "content": "Customer support message:\nI was charged twice for September, please refund one of the charges." } ],
  "scoring": { "queries": [
    { "id": "refund", "template": "\n\nIs the customer asking for a refund?\nAnswer:{?}", "options": [" yes", " no"] }
  ] }
}'
```

Read `results[0].queries[0].combinations[0].options`: `p` is the probability of each answer among the two, `floor_p`
the same with empty content. A decision is `p(" yes") >= threshold`, with the threshold chosen on a few labelled
messages ([Calibration](calibration.md)).

## 2. Two classifications in parallel

The content is read once; both questions are answered from that read, so the second question costs only its own
tokens. The first question is a list of topics named in the template; the second is a yes/no with the definition in the
question ([Writing questions](writing-questions.md)).

```bash
curl http://localhost:8000/v1/score -H "Content-Type: application/json" -d '{
  "model": "spinf/gemma-4-12b-fp8",
  "messages": [ { "role": "user", "content": "Customer support message:\nHi, since this morning'"'"'s SSO change nobody on our team can log in, and we have a board demo at 2pm. Please call me back." } ],
  "scoring": { "queries": [
    { "id": "topic",
      "template": "\n\nTopic (billing, technical, account, shipping, or cancellation):{?}",
      "options": [" billing", " technical", " account", " shipping", " cancellation"] },
    { "id": "urgent",
      "template": "\n\nIs this urgent (business blocked, security, legal threat, data loss, or a deadline within 24 hours)?\nAnswer:{?}",
      "options": [" yes", " no"] }
  ] }
}'
```

*Shape* of the result (one entry per question, the options with their probabilities):

```json
{ "input_id": "0", "queries": [
  { "id": "topic", "combinations": [ { "values": {}, "options": [
      { "text": " billing", "logprob": -5.1, "p": 0.02, "floor_p": 0.31 },
      { "text": " technical", "logprob": -1.2, "p": 0.91, "floor_p": 0.22 },
      { "text": " account", "logprob": -4.0, "p": 0.06, "floor_p": 0.19 },
      { "text": " shipping", "logprob": -7.9, "p": 0.00, "floor_p": 0.14 },
      { "text": " cancellation", "logprob": -7.1, "p": 0.01, "floor_p": 0.14 } ],
      "residual": 0.71, "floor_residual": 0.80 } ] },
  { "id": "urgent", "combinations": [ { "values": {}, "options": [
      { "text": " yes", "logprob": -0.4, "p": 0.88, "floor_p": 0.45 },
      { "text": " no", "logprob": -2.4, "p": 0.12, "floor_p": 0.55 } ],
      "residual": 0.33, "floor_residual": 0.36 } ] } ] }
```

Add as many questions as the job needs: intent, sentiment, a prompt-injection check, a policy check, each a question
in the same list. With a placeholder and `"combinations": "all"` one template asks a grid (every `{aspect}` × every
`{sentiment}`).

## 3. Classification plus extraction

Extraction slots let the model write the answer instead of choosing it: `{*}` for text, `{n}` for a number, `{$}` for
an amount with its currency. They go in the same `queries` list and are decoded after the scoring pass on the same read
of the content. A `{*}` slot is best written in quotes and named: the text after the slot (the closing quote) is where
the answer ends.

```bash
curl http://localhost:8000/v1/score -H "Content-Type: application/json" -d '{
  "model": "spinf/gemma-4-12b-fp8",
  "messages": [ { "role": "user", "content": "News:\nApple sold 3.5 million iPhones in the quarter, up from 3.1 million a year earlier, the company said on Tuesday." } ],
  "scoring": { "queries": [
    { "id": "dir", "template": "\n\nDid iPhone sales go{?}", "options": [" up", " down"] },
    { "id": "units", "template": "\n\nQ: How many iPhones were sold in the quarter?\nA:{n}" },
    { "id": "company", "template": "\n\nThe company in this news is \"{*}\"", "top_k": 3 }
  ] }
}'
```

*Shape* of the result. A scored question answers as above; an extraction answers with the most likely strings, merged
over spellings (numbers by value), each with its probability, plus `mass` (the probability on all answers found) and,
for numbers, `p_negative` (how much of the top value's magnitude sits on its negative, which matters for accounting
brackets):

```json
{ "input_id": "0", "queries": [
  { "id": "dir", "combinations": [ { "values": {}, "options": [
      { "text": " up", "logprob": -0.3, "p": 0.93, "floor_p": 0.52 },
      { "text": " down", "logprob": -2.9, "p": 0.07, "floor_p": 0.48 } ], "residual": 0.41, "floor_residual": 0.55 } ] },
  { "id": "units", "kind": "number", "combinations": [ { "values": {},
      "answers": [ { "text": " 3.5 million", "p": 0.81, "value": 3500000.0 }, { "text": " 3.1 million", "p": 0.09, "value": 3100000.0 } ],
      "mass": 0.94, "p_negative": 0.0 } ] },
  { "id": "company", "kind": "text", "combinations": [ { "values": {},
      "answers": [ { "text": "Apple", "p": 0.97 }, { "text": "Apple Inc.", "p": 0.01 } ], "mass": 0.99 } ] } ] }
```

Extraction slots have no empty floor. `max_tokens` (default 16, up to 64) caps the answer length, `top_k` (default 5,
up to 20) the number of answers returned. When the answers are a known short list, a scored question is cheaper and
has a floor; use a slot when the answer is open (a name, an amount, a date).

**Generation.** `"generate": true` on a `{*}` slot writes free text instead of searching short answers (a one-sentence
summary, a caption), up to 100 tokens, greedy, or with sampling settings
(`"generate": {"temperature": 0.7, "top_p": 0.9, "seed": 0, "n": 1}`); the answer comes back as `{text, finish}` with no
probability.

```json
{ "id": "summary", "template": "\n\nIn one sentence, the news is: \"{*}\"", "generate": true, "max_tokens": 40 }
```

## 4. Many contents in one call

Send `inputs` instead of `messages`. The queries in `scoring.queries` are shared by every input (or only by those that
list them in `query_ids`); an input can add its own. The empty floor of a shared question is computed once for the
call, and the inputs are packed into the same forward passes.

```bash
curl http://localhost:8000/v1/score -H "Content-Type: application/json" -d '{
  "model": "spinf/gemma-4-12b-fp8",
  "inputs": [
    { "id": "apple",
      "messages": [ { "role": "user", "content": "Apple reported record iPhone revenue while Mac margins slipped." } ] },
    { "id": "exxon",
      "messages": [ { "role": "user", "content": "Exxon cut output after OPEC+ extended its supply cuts." } ],
      "query_ids": ["sector"] }
  ],
  "scoring": { "queries": [
    { "id": "sector",
      "template": "\n\nThis article is mainly about the{?}",
      "options": [" technology sector", " energy sector", " banking sector"] },
    { "id": "dir",
      "template": "\n\nWill {company}'"'"'s {metric} rise or fall next quarter?\nAnswer:{?}",
      "options": [" rise", " fall"],
      "combinations": "all",
      "values": { "company": ["Apple"], "metric": ["revenue", "margin"] } }
  ] }
}'
```

The response (Gemma 4 12B, real numbers):

```json
{
  "results": [
    { "input_id": "apple",
      "queries": [
        { "id": "sector",
          "combinations": [
            { "values": {},
              "options": [
                { "text": " technology sector", "logprob": -8.439295, "p": 0.933484, "floor_p": 0.019076 },
                { "text": " energy sector", "logprob": -12.483436, "p": 0.016359, "floor_p": 0.41683 },
                { "text": " banking sector", "logprob": -11.363065, "p": 0.050157, "floor_p": 0.564094 } ],
              "residual": 0.999768, "floor_residual": 0.999993 } ] },
        { "id": "dir",
          "combinations": [
            { "values": { "company": "Apple", "metric": "revenue" },
              "options": [
                { "text": " rise", "logprob": -1.776795, "p": 0.801008, "floor_p": 0.525006 },
                { "text": " fall", "logprob": -3.169399, "p": 0.198992, "floor_p": 0.474994 } ],
              "residual": 0.788792, "floor_residual": 0.861084 },
            { "values": { "company": "Apple", "metric": "margin" },
              "options": [
                { "text": " rise", "logprob": -2.467582, "p": 0.564992, "floor_p": 0.595533 },
                { "text": " fall", "logprob": -2.729028, "p": 0.435008, "floor_p": 0.404467 } ],
              "residual": 0.849928, "floor_residual": 0.84605 } ] } ] },
    { "input_id": "exxon",
      "queries": [
        { "id": "sector",
          "combinations": [
            { "values": {},
              "options": [
                { "text": " technology sector", "logprob": -10.925335, "p": 0.010058, "floor_p": 0.019076 },
                { "text": " energy sector", "logprob": -6.343364, "p": 0.982723, "floor_p": 0.41683 },
                { "text": " banking sector", "logprob": -11.256995, "p": 0.007219, "floor_p": 0.564094 } ],
              "residual": 0.998211, "floor_residual": 0.999993 } ] } ] }
  ],
  "warnings": [],
  "usage": { "prompt_tokens": 112, "completion_tokens": 0, "total_tokens": 112,
             "prompt_tokens_details": { "content_tokens": 22, "query_tokens": 50, "floor_tokens": 40 } }
}
```

The `sector` floor is the same for both inputs: it was computed once. The article lifts " rise" for Apple's revenue
from 0.53 to 0.80; for its margin, `p` (0.56) sits near the floor (0.60), so the content says little about it. The
`sector` residuals are close to 1 because its answers have two tokens each; `p` still separates them clearly.

## 5. A prompt pack next to your own question

A pack is a ready-made, calibrated set of questions ([Prompt packs](prompt-packs.md)). It is scored in the same pass as
your questions and returns a 0 to 1 score per category and a decision.

```bash
curl http://localhost:8000/v1/score -H "Content-Type: application/json" -d '{
  "model": "spinf/gemma-4-12b-fp8",
  "messages": [ { "role": "user", "content": "Guaranteed 300% returns in 7 days!! DM me your wallet seed phrase." } ],
  "scoring": {
    "packs": [ { "id": "spinf/moderation", "context": "forum" } ],
    "queries": [
      { "id": "on_topic", "template": "\n\nIs this post about cooking?\nAnswer:{?}", "options": [" yes", " no"] }
    ]
  }
}'
```

Each result then has `packs` beside `queries`:

```json
{ "input_id": "0",
  "queries": [ { "id": "on_topic", "…": "…" } ],
  "packs": [ { "id": "spinf/moderation", "version": "0.1.3", "context": "forum",
    "categories": { "fraud": { "score": 0.93, "percentile": 1.0, "flagged": true },
                    "spam": { "score": 0.88, "percentile": 0.99, "flagged": null }, "…": "…" },
    "decision": { "unsafe": true, "mode": "micro_layer", "score": 1.21, "threshold": -0.23 } } ] }
```

## 6. Video frames and an extraction slot

A video is sent as frames you sampled, each with its time in seconds, in a `video` part; the engine never decodes a
video file (spinf-client has a helper that samples with ffmpeg). The frames stay where you put them in the content, so a
line of text can introduce them, and the questions can say "this commercial". Every question, scored or extracted,
shares the one read of the frames.

```bash
curl http://localhost:8000/v1/score -H "Content-Type: application/json" -d '{
  "model": "spinf/gemma-4-12b-fp8",
  "messages": [ { "role": "user", "content": [
    { "type": "text", "text": "A 30-second TV commercial:\n" },
    { "type": "video", "video": { "detail": "low", "frames": [
      { "t": 0.75, "image_url": { "url": "data:image/jpeg;base64,/9j/4AAQ…" } },
      { "t": 2.25, "image_url": { "url": "data:image/jpeg;base64,/9j/4AAQ…" } },
      { "t": 3.75, "image_url": { "url": "data:image/jpeg;base64,/9j/4AAQ…" } }
    ] } },
    { "type": "text", "text": "\n" }
  ] } ],
  "scoring": { "queries": [
    { "id": "funny", "template": "\nQ: Is this commercial trying to be funny?\nA:{?}", "options": [" Yes", " No"] },
    { "id": "celebrity", "template": "\nQ: Does this commercial feature a celebrity?\nA:{?}", "options": [" Yes", " No"] },
    { "id": "animals", "template": "\nQ: Are there animals in this commercial?\nA:{?}", "options": [" Yes", " No"] },
    { "id": "brand", "template": "\nThe brand featured in this commercial is \"{*}\"", "top_k": 3 }
  ] }
}'
```

The scored questions answer as in example 2, the brand as in example 3 (`answers` with `p` and `mass`). On a set of TV
commercials with the brand as gold, the top answer was right 0.97 of the time from 32 frames at `low` detail on the 12B
(0.99 on the 31B), 0.57 from 4 frames (0.74) and 0.19 from one frame (0.29): about 20 low-detail frames are enough for
what is on screen, and fewer lose fast. A low-detail frame is about 70 tokens plus its timestamp; 20 frames are about
1,500 tokens of content, read once for every question. Frames larger than their detail's size are resized by the
engine; sending them at about that size (a 16:9 frame at 528 x 288 is about 23 KB in base64) keeps the request small
(`--max-body-bytes`, [Configuration](configuration.md)).

## 7. A long document

An input may hold as much content as `--max-tokens` allows ([Configuration](configuration.md): 65,536 tokens on the 12B
by default, 32,768 on the 31B, 16,384 on the Qwen model; `--max-content-tokens` is the request-side cap, 4,096 unless
raised). A content over the packing budget runs alone in its own pass. The call is the same as example 1 with a longer
`content`; the cost grows faster than the length. Measured on the 12B on an RTX 4090, one scored question and one `{n}`
slot on the content:

| content | time | GPU memory |
|---|---|---|
| 8k tokens | 2.0 s | 13.5 GB |
| 16k | 2.9 s | 14.5 GB |
| 32k | 8.2 s | 16.5 GB |
| 58k | 23.8 s | 19.9 GB |

The slot found a code placed at the end of the text every time; a scored read of a long document, though, reacts mostly
to the text's end. For a document-level read, cut the document at natural boundaries (sections, pages, a few thousand
tokens each), score the passages as `inputs` of one call, and aggregate the scores ([Calibration](calibration.md)):
the passages pack into the same passes, so it costs about the same as one long input and reads every part of the
document.
