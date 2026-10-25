# Decision models: Gemma 4 as a calibrated decision model, at scale

A **decision model** is an AI model used to hand software a decision rather than a paragraph. You give it a state (a
message, a document, a record from your application) and the options you allow, and it returns a choice or, better, a
probability for each option. There is no text to generate, parse or validate: the output is one of your options, with
a number your code can act on.

The term has older meanings too. In business process management, **Decision Model and Notation (DMN)** is a standard for
writing decision rules as tables and diagrams, and in research and management a decision model is often a **decision
tree** that maps choices, outcomes and their probabilities. This guide is about the AI sense: a language model used to
decide, not to write.

## A leading open-weights model, used as a decision model

spinf-serve runs Gemma 4, Google DeepMind's open-weights model family, on one GPU, optimized to be used as a decision
model at very large scale:

- **You define the decision in plain language.** A question and the options you allow, written as a template. No
  training run, no labelled dataset to start, and no new model to adopt for each new decision.
- **You get a probability for every option.** Normalised over your options, so they sum to 1. The answer is always one
  of yours.
- **The probabilities come with a calibration check.** Each question is also scored with empty content, so you can see
  how much the state itself moved each option.
- **One read, many decisions.** The state is read once and every question is answered from that read, so a second or a
  hundredth decision about the same state costs only its own tokens.

Because the model is a general one, the same call can take a routing decision, a policy check and a risk flag about
the same input, and a new decision is a new question, not a new model. The default model is the 12B
(`spinf/gemma-4-12b-fp8`, one 24 GB card); the larger 31B (`spinf/gemma-4-31b-fp8`) needs a 48 GB card. spinf-serve
also runs Qwen3.8-27B (`spinf/qwen3.8-27b-int4`) and Perplexity's pplx-decider (`spinf/pplx-decider-v1-27b-int4`, a
model trained to answer closed questions with a code); the table is in [Configuration](../configuration.md#models).
The same request runs unchanged against the hosted API at api.spinf.com with a key.

## An example: a decision on application state

The state does not have to be prose. Here it is an order record, and the decision is what to do next:

```json
{
  "model": "spinf/gemma-4-12b-fp8",
  "messages": [{ "role": "user", "content": "Order:\n{\"order_id\": \"A-10482\", \"amount_usd\": 1840, \"customer_since\": \"2026-09-27\", \"previous_orders\": 0, \"shipping_country\": \"US\", \"billing_country\": \"BR\", \"items\": [\"gift card x4\"], \"payment\": \"new card, 3 declined attempts before success\"}" }],
  "scoring": { "queries": [{
    "id": "action",
    "template": "\n\nNext action (ship, hold for review, or cancel):{?}",
    "options": [" ship", " hold for review", " cancel"]
  }]}
}
```

Sent to `http://localhost:8000/v1/score`, the 12B with no examples returned:

| Option | `p` | `floor_p` (no order) | Lift |
|---|---|---|---|
| ship | 0.59 | 0.75 | −0.16 |
| hold for review | 0.30 | 0.04 | +0.26 |
| cancel | 0.11 | 0.22 | −0.11 |

The largest raw probability is still "ship", because the question alone leans heavily that way. The floor shows what
the order itself says: it pushes "hold for review" up from 0.04 to 0.30 and pushes the other two down. This is why a
decision model should return calibrated probabilities rather than a single label: the label would have been "ship".
Read the lift, or rank each score against your own history ([Calibration](../calibration.md)), and set thresholds on
it ([probabilistic if statements](probabilistic-if.md)). A few labelled orders from your history in the content
sharpen the scores further ([few-shot classification](zero-shot-few-shot-classification.md)).

## Measured on common decisions

On the test sets of our public benchmark, the 12B with a few labelled examples in the prompt:

- **Is this ticket urgent?** 92% correct (8 examples, 130 synthetic tickets).
- **Can a bot resolve it?** 93% correct (8 examples, same tickets).
- **Does this post break the policy?** 99% correct (4 examples, 130 synthetic posts).
- **Is this message a prompt injection?** 92% correct (4 examples, calibrated, on the public deepset/prompt-injections test split).

Synthetic sets are cleaner than real data: measure on a labelled sample of your own decisions before you rely on the
thresholds. The sets, questions and code are on [GitHub](https://github.com/spinfinc/spinf-benchmarks).

## Decisions at massive scale

The cost of a decision is the tokens the engine processes: the state once, plus each question's own tokens. A decision
on a 100-token record with one 20-token question is about 120 tokens. The 12B processes about 10,000 tokens per second
on one RTX 4090 (the content once, then each question's own tokens), so **a million such decisions are about 120
million tokens, or about 3.5 hours on one card**, sent with many records per call. A second question about the same
record adds its own 20 tokens, not another read of the record. [Scoring without output tokens](no-output-tokens.md)
shows the arithmetic.

This makes it practical to decide on everything rather than a sample: every order, every ticket, every post, every
record in an archive, and to re-run the whole history when a policy or a threshold changes.

## Bulk and real time

The engine is built for throughput: backfills, backlogs, nightly sweeps, and measuring a new decision on your history
before it goes live, with many records per call. A decision inside a live flow (routing a ticket as it arrives,
holding an order at checkout) runs the same questions and thresholds one record at a time; a server you run yourself
answers a single short record in well under a second once warm, and a second server on another card adds capacity.
Set the thresholds on your history first, then wire the live path to the same questions.

## Try it

The [examples](../examples.md) run several decisions about one message against your own server; the
[quickstart](../quickstart.md) is the first call. Next: [probabilities from a language model](llm-probabilities.md).

Gemma is a trademark of Google LLC.
