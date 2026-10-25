"""Case-insensitive options: an answer word is read as the sum of its case and spacing variants.

A model spreads an answer over its spellings: after "Answer:" Gemma 4 12B puts ~0.23 on " Yes", ~0.018 on " yes", ~0.006 on
" YES" and ~0.001 on "Yes" (research/case/case_probe.py, fixture articles x 12 statements). Reading one spelling makes the score
depend on the case the model happens to prefer; summing every variant makes the option case-insensitive. Variants that are one
token share the option's read row, so they cost nothing but lm-head columns.
"""


def case_variants(tok, word: str, single_token_only: bool = True) -> list[str]:
    """the spellings of `word` to sum: lower / Capitalized / UPPER x with and without a leading space, in that order, without
    duplicates (e.g. "I"); with single_token_only, only those the tokenizer keeps as one token (the others would need branches
    of their own). The word as given is always kept, even when it is several tokens."""
    base = word.strip()
    forms = []
    for f in (base.lower(), base[:1].upper() + base[1:].lower(), base.upper()):
        for s in (" " + f, f):
            if s not in forms:
                forms.append(s)
    out = [word]
    for s in forms:
        if s in out:
            continue
        if single_token_only and len(tok(s, add_special_tokens=False)["input_ids"]) != 1:
            continue
        out.append(s)
    return out
