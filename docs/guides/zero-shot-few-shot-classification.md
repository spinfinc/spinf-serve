# Zero-shot and few-shot classification with a language model, without fine-tuning

A language model can classify text without a training run. You describe the task in plain language (zero-shot), or
you also show it a few labelled examples in the prompt (few-shot, also called in-context learning). No model is
trained, and changing the labels is a matter of editing a question.

This guide explains both approaches as spinf-serve runs them, and what a few examples change, measured on our public
benchmark. The same requests run unchanged against the hosted API at api.spinf.com with a key.

## Zero-shot: the labels are the question

In zero-shot classification the model sees only the content and a question that names the possible answers. With
spinf-serve, the question is a template that ends where the answer goes, and the answers are the options to score:

```json
{
  "id": "topic",
  "template": "\n\nTopic (billing, technical, account, shipping, or cancellation):{?}",
  "options": [" billing", " technical", " account", " shipping", " cancellation"]
}
```

The engine returns the probability of each option, and they sum to 1. Nothing is generated for a scored question, so
there is no free text to parse and no label outside your list. It also returns the floor of each answer: the same
question asked with empty content. Reading the lift over the floor removes the question's own lean towards one answer,
which often helps zero-shot questions with several answers (see [Calibration](../calibration.md)).

Zero-shot works best when the labels are ordinary words with a clear meaning: sentiment, a yes/no fact, a short list
of distinct topics.

## Few-shot: labelled examples in the prompt

Few-shot classification adds a handful of labelled examples to the content, before the item to score. Each example
is written in the same format as the question, with its answer. The examples show the model your labels, where your
boundaries are, and the edge cases your team has already decided.

In-context learning is not fine-tuning: the model's weights do not change. The examples are part of the input of each
call, so you can change them, version them and test them on your own data at any time. They are also read with every
input, so keep them short and send many items per call. [Writing questions](../writing-questions.md) shows the format.

## What a few examples change, measured

These figures come from our public benchmark, on the 12B (`spinf/gemma-4-12b-fp8`, Gemma 4 12B as spinf-serve runs
it). "No examples" is zero-shot, and "calibrated" means it was read as the lift over the floor. "With examples" is 4 or
8 labelled examples in the prompt, as a mean over three different example sets. The examples are never the items being
scored.

| Task | Test set | No examples | With examples |
|---|---|---|---|
| Sentiment toward each company in a news article | 255 mentions, synthetic | 76% | 95% (8) |
| Support ticket: can a bot resolve it? | 130 tickets, synthetic | 75% (calibrated) | 93% (8) |
| Moderation: policy category (7) | 130 posts, synthetic | 79% | 98% (8) |
| Review: sentiment per aspect | 625 labels, synthetic | 84% (calibrated) | 94% (8) |
| Prompt injection | 116 messages, public (deepset/prompt-injections) | 78% | 92% (4, calibrated) |

The synthetic sets were written for the benchmark, labelled by a language model and checked blind by a second model.
They are cleaner than real data, so expect lower figures on your own content. The test sets, the questions, the
example sets and the code to re-run everything are public:
[spinf-benchmarks on GitHub](https://github.com/spinfinc/spinf-benchmarks).

The pattern is consistent. Examples help most where the labels depend on your own definitions: which tickets a bot
can resolve, which of seven policy categories a post falls into, how a review treats one aspect among several.

## When zero-shot is already enough

Some questions need no examples at all:

- **Clear sentiment.** On 300 public product reviews, overall polarity was 97% correct with no examples (calibrated).
- **Plain facts.** On synthetic news, "does the article report a guidance change?" was 98% correct with no examples,
  and 99% with 8.

Start zero-shot. Add examples only for the questions where the scores do not separate your labels well enough.

## A practical workflow

1. **Label a sample.** 100–300 items of your own data, with the answers your team would give.
2. **Score zero-shot**, with two or three wordings of each question. Read the lift over the floor for questions with
   several answers.
3. **Add 4, then 8 examples** to the questions that need them. Cover every answer at least once, and include the
   cases people get wrong. Try more than one example set.
4. **Keep the best version fixed.** Scores compare over time only under the same template, answers, examples and
   model build (the response's `system_fingerprint` names the build).
5. **Choose thresholds** on the probabilities, for what is automated and what goes to a person. See
   [Probabilistic if statements](probabilistic-if.md).

A round on 300 items with 8 examples is about 400,000 tokens: under a minute on one RTX 4090 with the 12B (about
10,000 processed tokens per second), so trying several variants costs little.

## Try it

- The [examples](../examples.md) show the few-shot format in a call against your own server; the
  [quickstart](../quickstart.md) is the first call.
- Related guides: [Probabilities from a language model](llm-probabilities.md) and
  [Probabilistic if statements](probabilistic-if.md). For the example format, see
  [Writing questions](../writing-questions.md).
