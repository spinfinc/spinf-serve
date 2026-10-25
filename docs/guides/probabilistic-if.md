# Probabilistic if statements: decisions in code with LLM confidence thresholds

Most automation built on language models ends in an `if`. Is this ticket about billing? Does this post break the
policy? Does this article report a guidance change? When the model returns text, the `if` has to trust a label.
When it returns a probability, the `if` can say how sure it needs to be:

> Not "if billing, route to billing", but "if billing is at least 80% likely, route to billing; if it is
> between 40% and 80%, ask a person".

This is probabilistic programming in its most practical form: every branch has a confidence threshold, and the
uncertain middle goes to a human in the loop.

## The pattern

spinf-serve returns, for each question, the probability of each answer you list ([Probabilities from a language
model](llm-probabilities.md)). A decision is then a comparison with two thresholds:

- **above the upper threshold**: act automatically;
- **between the two**: the review band, sent to a person;
- **below the lower threshold**: take the default path.

A minimal version in Python, with one yes/no question, against a server on the same machine (the same body goes to
`https://api.spinf.com/v1/score` with an `Authorization: Bearer` header for the hosted API):

```python
import requests

def p_yes(text: str, question: str) -> float:
    resp = requests.post(
        "http://localhost:8000/v1/score",
        json={
            "model": "spinf/gemma-4-12b-fp8",
            "messages": [{"role": "user", "content": "Customer support message:\n" + text}],
            "scoring": {"queries": [{
                "id": "q",
                "template": f"\n\n{question}\nAnswer:{{?}}",
                "options": [" yes", " no"],
            }]},
        },
        timeout=60,
    )
    resp.raise_for_status()
    options = resp.json()["results"][0]["queries"][0]["combinations"][0]["options"]
    return next(o["p"] for o in options if o["text"] == " yes")

p = p_yes(ticket, "Is this a billing question (charges, refunds, invoices or payment methods)?")
if p > 0.8:
    route(ticket, "billing")
elif p > 0.4:
    review(ticket)  # a person decides, and the decision becomes a labelled example
else:
    route(ticket, "general")
```

Each option in the response also has `floor_p`, the same question scored with empty content. When a question leans
towards one answer, compare against the lift `p − floor_p` rather than `p` alone ([Calibration](../calibration.md)).

For a backlog, send many items per call with `inputs` instead of one call per item: the items are packed into the
same forward passes and the floor of each question is computed once for the call. A live flow, such as routing tickets
as they arrive, runs the same questions and thresholds one item at a time against the same server.

## Choosing the thresholds: precision and recall on your history

Thresholds are a trade-off between how much you automate and how often an automatic decision is wrong. Set them on
data you already have:

1. Take a few hundred past items with the decision your team made (the label).
2. Score them with the question you plan to use.
3. For each candidate threshold, count the items above it (what you would automate) and how many of those carry a
   different label (the errors). That is precision and recall for the automatic branch.
4. Pick the upper threshold where the error rate is acceptable for the action, and the lower one where the items
   below it are safe to send down the default path.

Stricter actions deserve stricter thresholds. Removing a post or refunding a charge might need 0.95; tagging a
ticket for a queue can live with 0.7. Each question can have its own thresholds, and a question can have several
branches: route to the most likely queue when its probability is high enough, otherwise review.

Scoring 300 labelled items with a few questions is a few hundred thousand tokens, under a minute on one card, so you
can try several wordings and example sets before choosing ([Writing questions](../writing-questions.md)).

## Why the review band matters

Accuracy figures describe averages; thresholds let you decide where the errors go. In our benchmark on 130 synthetic
support tickets, with 8 labelled examples in the prompt, "is this urgent?" was 92% correct, and an urgent ticket
outranked a non-urgent one 98% of the time (AUC 0.98). A good ranking means most mistakes sit near the middle of the
scale: a review band catches them, and the clear cases at both ends can be automated.

The band also improves the system. Every item a person reviews is a new labelled example: add the hard ones to the
examples in the prompt, and re-score a sample of recent items every week to check that the thresholds still hold.

## Many questions, many branches

Because the content is read once and each extra question costs only its own tokens, one call can feed several
`if` statements: urgency, topic, "can a bot resolve it?", "does it break the policy?". Combine them in code the way
you would combine any other signals, for example routing on the topic only when the message is not urgent.

## Try it

The [examples](../examples.md) show two classifications and an extraction about one message in a single call, and the
[quickstart](../quickstart.md) the first call against your own server. Related: [intent classification](intent-classification.md)
and [Calibration](../calibration.md).
