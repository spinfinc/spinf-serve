# Prompt packs for spinf-serve

A **prompt pack** is a ready-made, calibrated set of questions for one job, published as a versioned file. A server
preloads it; a request names it (`"packs": [{"id": "spinf/moderation"}]`) and gets the pack's category scores and
decision back next to its own questions, from the same read of the content. Packs are how specialised knowledge travels
around spinf-serve: anyone can write one for any model the engine serves and publish it from their own GitHub repository.

This page lists every pack we know of, ours and others'. **If you publish one, [tell us](#list-your-pack) and it is added.**

```sh
spinf-serve serve spinf/gemma-4-12b-fp8 --packs github:thinkingtext/spinf-news-pack@0.1.0   # one source per --packs
spinf-serve packs list --packs github:thinkingtext/spinf-news-pack@0.1.0 --model spinf/gemma-4-12b-fp8
curl -s localhost:8000/v1/packs                                                              # what this server enables
```

A pack is written for one model; the server enables the packs whose model it serves. The official `spinf/*` packs load
by default and are also served by the hosted API at `api.spinf.com`; every other source is yours to add with `--packs`
(a GitHub repository at a tag, or a local folder). Format, request fields and sources: [docs/prompt-packs.md](docs/prompt-packs.md).

## spinfinc / spinf-prompts

The official packs, maintained with the engine: [github.com/spinfinc/spinf-prompts](https://github.com/spinfinc/spinf-prompts). Loaded by default.

| pack | job | model |
|---|---|---|
| [`spinf/moderation`](https://github.com/spinfinc/spinf-prompts/tree/main/packs/moderation) | content moderation: 22 categories of user-generated text, percentile on normal traffic, safe / unsafe decision | `spinf/gemma-4-12b-fp8` |
| [`spinf/triage`](https://github.com/spinfinc/spinf-prompts/tree/main/packs/triage) | inbound emails, chats and tickets: route, intent, priority, tone, risk, personal data, phishing; needs-attention decision | `spinf/gemma-4-12b-fp8` |
| [`spinf/products`](https://github.com/spinfinc/spinf-prompts/tree/main/packs/products) | product listings and feeds: Shopify taxonomy category, attributes, kind of offer, quality, claims, restricted products; needs-review decision | `spinf/gemma-4-12b-fp8` |
| [`spinf/reviews`](https://github.com/spinfinc/spinf-prompts/tree/main/packs/reviews) | reviews and customer feedback: opinion and implied stars, 16 aspects praised or criticised, requests, intent, tone, risks; needs-follow-up decision | `spinf/gemma-4-12b-fp8` |
| [`spinf/brand`](https://github.com/spinfinc/spinf-prompts/tree/main/packs/brand) | brand perception on a web page or search result, with a brand parameter: 20 perception items, intents, page type, audience; brand-alert decision | `spinf/gemma-4-12b-fp8` |
| [`spinf/entertainment`](https://github.com/spinfinc/spinf-prompts/tree/main/packs/entertainment) | synopses of films, series, books, games, podcasts: genres, mood, themes, advisories, audience; restricted-to-adults decision | `spinf/gemma-4-12b-fp8` |
| [`spinf/moderation-qwen`](https://github.com/spinfinc/spinf-prompts/tree/main/packs/moderation-qwen) | content moderation for the Qwen model | `spinf/qwen3.8-27b-int4` |
| [`spinf/moderation-qwen-lite`](https://github.com/spinfinc/spinf-prompts/tree/main/packs/moderation-qwen-lite) | one overall unsafe score from three questions | `spinf/qwen3.8-27b-int4` |
| [`spinf/moderation-decider`](https://github.com/spinfinc/spinf-prompts/tree/main/packs/moderation-decider) | content moderation in the decision model's own question format | `spinf/pplx-decider-v1-27b-int4` |

## thinkingtext / spinf-news-pack

The question sets behind Thinking Text's news indexes: [github.com/thinkingtext/spinf-news-pack](https://github.com/thinkingtext/spinf-news-pack).
Report-only packs (standardised scores and percentiles per slot, no decision). `--packs github:thinkingtext/spinf-news-pack@0.1.0`.

| pack | job | model |
|---|---|---|
| [`thinkingtext/oil`](https://github.com/thinkingtext/spinf-news-pack/tree/main/packs/oil) | an oil-market article: demand, supply, inventories, OPEC+, disruption risk, refining margins, macro context; a signed six-slot composite | `spinf/gemma-4-12b-fp8` |
| [`thinkingtext/market`](https://github.com/thinkingtext/spinf-news-pack/tree/main/packs/market) | business news: the general market read, 126 macro and market cells signed toward higher US stock prices | `spinf/gemma-4-12b-fp8` |
| [`thinkingtext/equity-<ticker>`](https://github.com/thinkingtext/spinf-news-pack/tree/main/packs) | an article about one company: demand, profits, outlook, competition, regulation, costs, share price and valuation, results against expectations; twelve names (`aapl`, `amd`, `amzn`, `csco`, `googl`, `intc`, `meta`, `msft`, `mu`, `nvda`, `pltr`, `tsla`) | `spinf/gemma-4-12b-fp8` |

## List your pack

Written a pack? [Open an issue](https://github.com/spinfinc/spinf-serve/issues/new?template=pack.yml) with the
repository link, the pack ids, the model and a line on what it does, and it goes on this page. The only requirements:
a public repository with a `prompt-packs.json` at its root that `spinf-serve packs validate` accepts, your own
namespace (`spinf/` is reserved for the official packs), and a licence. How to write one: [docs/prompt-packs.md](docs/prompt-packs.md),
the [pack format](https://github.com/spinfinc/spinf-prompts/blob/main/docs/pack-format.md) and the
[`spinf-packs` agent skill](https://github.com/spinfinc/skills).
