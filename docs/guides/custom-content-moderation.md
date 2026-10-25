# Custom content moderation policies with a language model

Every platform has rules that a generic moderation list does not cover: counterfeit goods on a marketplace, off-topic
posts in a professional community, medical claims in product reviews, recruiting scams in a job board. Trust and safety
teams end up with a policy document that only people can apply, and a queue that grows faster than the team.

spinf-serve scores content on your own GPU with an open-weights language model. For moderation, that means a policy
classifier you write yourself: your rules as questions, a few examples per category, and a probability for each rule
on every post. This guide covers how to write the policy, set thresholds per category, clear a moderation backlog and
moderate live content. The same call runs unchanged against the hosted API at api.spinf.com with a key.

## Write the policy as questions

A custom content moderation policy becomes two kinds of questions:

- **A yes/no question for "does it break the policy?"**, with the policy summarized in the question: "Does this post
  break the platform's content policy (harassment, hate, sexual content, violent threats, self-harm, spam or scams)?".
- **A closed question for the category**, with your moderation categories as the answers: "Policy category (safe,
  harassment, hate, sexual, violence, self-harm, or scam):".

```bash
curl http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spinf/gemma-4-12b-fp8",
    "messages": [ { "role": "user", "content": "User post:\nAnyone else think the new mods are power-tripping? Asking for a friend." } ],
    "scoring": {
      "queries": [
        { "id": "breaks_policy",
          "template": "\n\nDoes this post break the platform'\''s content policy (harassment, hate, sexual content, violent threats, self-harm, spam or scams)?\nAnswer:{?}",
          "options": [" yes", " no"] },
        { "id": "category",
          "template": "\n\nPolicy category (safe, harassment, hate, sexual, violence, self-harm, or scam):{?}",
          "options": [" safe", " harassment", " hate", " sexual", " violence", " self-harm", " scam"] }
      ]
    }
  }'
```

You can add a question for any rule of your own ("Does this listing sell counterfeit goods?", "Does this review make a
medical claim?"), and ask all of them of the same post in one call. The post is read once, and each extra question
costs only its own tokens.

Start each post with a short label, such as `User post:` or `Listing:`, so the model knows what it is reading. The
answers always come back as one of the options you listed, with its probability: there is no generated text to parse.

The official `spinf/moderation` prompt pack is the ready-made starting point: 22 categories asked through 106 questions,
calibrated thresholds and a safe / unsafe decision, fitted on the 12B. Name it in the request and it runs in the same
call and the same read as your own questions, so a platform rule the pack does not know sits next to the generic ones
([prompt packs](../prompt-packs.md)).

## Teach the edge cases with examples

A policy is mostly its edge cases: the joke that is not harassment, the polite message that is a scam. Put a few
labelled posts in the content before the post to score, covering each category and the cases your moderators argue
about. No training run and no model to host.

On 130 synthetic posts across six policy categories and safe content, including edgy-but-safe and polite-but-harmful
posts, the 12B (`spinf/gemma-4-12b-fp8`) got:

| Question | No examples | With examples |
|---|---|---|
| Breaks policy? (yes/no) | 92% correct | 99% correct, with 4 examples |
| Policy category (7) | 79% correct | 98% correct, with 8 examples |

The posts, questions, example sets and code are public in
[spinf-benchmarks](https://github.com/spinfinc/spinf-benchmarks). Real posts are harder than synthetic ones: teach your
policy with examples from your own platform, and measure on a labelled sample of your own content before relying on
the numbers.

## Set thresholds per category

A probability per rule lets you act differently on different risks:

- **Remove** what is clearly over the line: a high threshold, set so that almost nothing removed would be reversed on
  appeal.
- **Queue for review** the borderline band below it, where a moderator decides.
- **Allow** the rest.

Set each threshold on your own history: score a few hundred posts your team has already decided, and pick the cut-off
per category where the automatic decisions match your moderators closely enough. A stricter category (threats,
self-harm) can have a lower threshold for review than a milder one (off-topic). The
[probabilistic if statements guide](probabilistic-if.md) covers the mechanics.

## Clear the moderation backlog

A backlog, or a periodic sweep of everything published last week, is a throughput job: send posts in batches with many
posts per call (`inputs`, [scoring API](../scoring-api.md#request-many-contents)), compare the scores with past
decisions, and work through the queue from the highest probability down.

Scoring every post is affordable, not just a sample, because the content is read once for all its questions. The 12B
processes about 10,000 tokens per second on one RTX 4090 (the content once, plus each question's own tokens): a million
300-token posts with a handful of questions are about 300 million tokens, or about 8 hours on one card. Eight labelled
examples in front of every post add their tokens to every read (roughly 2,000 more per post, or seven to eight times
the work); keep them short, and keep the example set for the categories where it moves the numbers.

## Moderate live posts

To check posts before they are published, run the same questions and thresholds against the same server. The engine
scores one request at a time per GPU with a small queue in front, so a card dedicated to moderation gives steady latency
(a single post with a few questions answers in well under a second on the 12B), and a second card takes the backlog
sweeps ([throughput](../configuration.md#throughput)). A good path is to calibrate the thresholds on your history in
bulk first, then point the live check at the same questions.

## Good to know

- **Other formats:** the 12B reads images, audio and video frames in the same call; the 31B and the Qwen model read
  images and video frames ([images, video and audio](../scoring-api.md#images-video-and-audio)).
- **Data:** with spinf-serve the content never leaves your machine.
- **Other checks in the same call:** classify the intent of a message, or check it for prompt injection, while you
  moderate it: each is one more question in the same list.

## Try it

The [quickstart](../quickstart.md) runs a first call; the [examples](../examples.md) show several classifications in one
call and a prompt pack next to your own questions. Related: [probabilistic if statements](probabilistic-if.md) and
[zero-shot and few-shot classification](zero-shot-few-shot-classification.md).
