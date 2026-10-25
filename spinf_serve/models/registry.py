"""family name -> stack (plan §4): the manifest's `family` when the engine runs one, else the family whose detect() recognises the
checkpoint's config.json (a Hub checkpoint served by its path)."""

from spinf_serve.models.base import Stack


def stacks() -> dict[str, type[Stack]]:
    from spinf_serve.models.gemma4 import Gemma4Stack
    from spinf_serve.models.qwen35.qwen import Qwen35Stack

    return {"gemma4": Gemma4Stack, "qwen35": Qwen35Stack}


def stack_for(model_dir: str, family: str = "") -> Stack:
    all_stacks = stacks()
    if family:
        if family not in all_stacks:
            raise ValueError(f"unknown model family {family!r}: one of {', '.join(all_stacks)}")
        if not all_stacks[family].detect(model_dir):
            raise RuntimeError(f"{model_dir}: not a {family} checkpoint (the manifest says it is)")
        return all_stacks[family]()
    for cls in all_stacks.values():
        if cls.detect(model_dir):
            return cls()
    raise RuntimeError(f"{model_dir}: no supported architecture family ({', '.join(all_stacks)}) recognises its config.json")
