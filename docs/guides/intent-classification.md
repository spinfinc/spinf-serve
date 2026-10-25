# Intent classification and ticket routing with confidence thresholds

Support teams sort every incoming ticket, chat and email: which queue it belongs to, whether it is urgent, whether a
bot or a help article can resolve it. Intent classification (or intent detection) does this sorting automatically,
and support ticket triage works when the automation knows when it is unsure.

spinf-serve scores messages on your own GPU with an open-weights language model. For each message it returns the
probability of each answer to your routing questions, so you can automate the clear cases and send the uncertain ones
to a person, at a confidence threshold you choose. The same call runs unchanged against the hosted API at
api.spinf.com with a key.

## Intents as answer options

Each routing decision is a closed question, and the intents, queues or categories are its answers:

```json
{
  "id": "intent",
  "template": "\n\nIntent (login problem, outage, billing, cancellation, or feature request):{?}",
  "options": [" login problem", " outage", " billing", " cancellation", " feature request"]
}
```

Add the other decisions as questions on the same message: "Is this urgent (business blocked, security, legal threat,
data loss, or a deadline within 24 hours)?", "Can a bot resolve this?", "Does it need a human agent?". The message is
read once, and each extra question costs only its own tokens:

```bash
curl http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spinf/gemma-4-12b-fp8",
    "messages": [ { "role": "user", "content": "Customer support message:\nI was charged twice for September and I need this fixed before payroll runs tomorrow." } ],
    "scoring": {
      "queries": [
        { "id": "intent",
          "template": "\n\nIntent (login problem, outage, billing, cancellation, or feature request):{?}",
          "options": [" login problem", " outage", " billing", " cancellation", " feature request"] },
        { "id": "urgent",
          "template": "\n\nIs this urgent (business blocked, security, legal threat, data loss, or a deadline within 24 hours)?\nAnswer:{?}",
          "options": [" yes", " no"] },
        { "id": "bot",
          "template": "\n\nCan a bot resolve this without a human agent?\nAnswer:{?}",
          "options": [" yes", " no"] }
      ]
    }
  }'
```

The answer is always one of your labels, with its probability: no free text to parse and no invented intents. Start
each message with a short label, such as `Customer support message:` or `Customer email:`, so the model knows what it
is reading. The same approach works for email classification and for chat messages. The official `spinf/triage` prompt
pack asks a ready-made set of these questions (route, specific intent, priority, tone, risk, personal data, phishing)
and runs in the same call as your own ([prompt packs](../prompt-packs.md)).

## Accuracy, measured

We measured ticket classification on 130 synthetic support tickets, labelled by a language model and checked blind by
a second model, with no examples and with 8 labelled example tickets placed in the prompt (never the tickets being
scored), on the 12B (`spinf/gemma-4-12b-fp8`):

- **Topic (5 topics):** 93% correct with 8 examples; 87% with no examples (calibrated). The 31B
  (`spinf/gemma-4-31b-fp8`) gets 95% with no examples.
- **Can a bot resolve it?** 93% correct with 8 examples; 75% with no examples (calibrated).
- **Urgent?** 92% correct with 8 examples; 85% with no examples. An urgent ticket outranks a non-urgent one 98% of the
  time (AUC 0.98).
- **Needs a human?** 87% correct with 8 examples, and 87% with no examples.

The tickets, questions, example sets and code are public in
[spinf-benchmarks](https://github.com/spinfinc/spinf-benchmarks). Synthetic tickets are cleaner than real ones: expect
lower numbers on your data, and measure them on your own history before going live.

The examples are the biggest lever. Put 4 to 8 of your own labelled tickets in the content, covering every intent and
the cases your team gets wrong, and the scores move closer to your team's decisions. No training run is needed. See
[zero-shot and few-shot classification](zero-shot-few-shot-classification.md).

## Choose a confidence threshold

A probability per intent turns routing into a threshold decision:

1. Score a labelled sample of past tickets (a few hundred is usually enough).
2. For each question, find the probability above which the automatic decision matches your team closely enough.
3. Above the threshold, route automatically; below it, send the ticket to a person.

You can use different thresholds per queue: a strict one for tickets a bot will answer on its own, a looser one for
tickets that only change their queue. The [probabilistic if statements guide](probabilistic-if.md) covers thresholds,
the review band and how to measure the trade-off.

## Many intents: two steps

Intent classification works best with a manageable list: a few to a few dozen intents, each named clearly. For a long
list, use hierarchical intents and ask two questions: first the broad category ("billing, technical, account, shipping,
or cancellation"), then the intent within that category, with a few labelled examples for each. Measure both steps on
a labelled sample of your own messages.

## Connect it to your helpdesk

spinf-serve is an HTTP server. Call it from your helpdesk's webhooks or automation rules (for example in Zendesk,
Intercom or Freshdesk) with the ticket text, and write the answer back as a tag, a priority or a queue. Ticket content
never leaves your machine.

## Calibrate first, then route live

1. **Calibrate on your history.** Score last year's tickets in bulk, many tickets per call (`inputs`,
   [scoring API](../scoring-api.md#request-many-contents)), compare with what your team decided, and pick your
   thresholds. The 12B processes about 10,000 tokens per second on one RTX 4090 (the content once, plus each
   question's own tokens): a million 300-token tickets with a few questions are about 300 million tokens, or about 8
   hours on one card; 8 labelled examples in front of every ticket add roughly 2,000 tokens to each read, so budget
   seven to eight times that.
2. **Go live against the same server.** The engine scores one request at a time per GPU with a small queue in front, so
   a card dedicated to routing answers a single ticket in well under a second and keeps that latency steady; bulk
   re-scoring runs on another card or off-peak ([throughput](../configuration.md#throughput)).
3. **Keep checking.** Re-score a sample of new tickets in bulk every week to catch drift.

## Try it

The [quickstart](../quickstart.md) runs a first call; the [examples](../examples.md) show several classifications in one
call. Related: [probabilistic if statements](probabilistic-if.md) and
[probabilities from a language model](llm-probabilities.md).
