"""Shared chat prompting and batched next-token label scoring."""
from __future__ import annotations

from typing import Iterable, Sequence


def chat_prompt(teacher, system: str, user: str) -> str:
    """Render a non-thinking chat prompt with a compatibility fallback."""
    try:
        return teacher.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return teacher.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )


def token_ids(tokenizer, values: Iterable[str]) -> list[int]:
    """Return first-token ids only for values encoded as exactly one token."""
    ids = []
    for value in values:
        encoded = tokenizer.encode(value, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(encoded[0])
    return ids


def digit_ids(tokenizer, digits: Iterable[int]) -> list[list[int]]:
    return [
        token_ids(tokenizer, (str(digit), " " + str(digit)))
        for digit in digits
    ]


def line_context(text: str, start: int, end: int, line_limit: int = 300) -> str:
    """Current line plus the preceding non-empty line as a short section hint."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end][:line_limit]
    previous = ""
    cursor = line_start - 1
    while cursor > 0:
        prev_start = text.rfind("\n", 0, cursor) + 1
        prev_end = text.find("\n", prev_start)
        if prev_end == -1:
            prev_end = len(text)
        candidate = text[prev_start:prev_end].strip()
        if candidate:
            previous = candidate[:80]
            break
        cursor = prev_start - 1
    return f"[mục: {previous}] {line}" if previous else line


def batch_group_scores(
    teacher,
    prompts: Sequence[str],
    label_id_groups: Sequence[Sequence[int]],
    batch_size: int,
    max_length: int,
) -> list[list[float]]:
    """Log-sum-exp score each target-token group from one next-token forward."""
    import torch

    device = next(teacher.model.parameters()).device
    results = []
    for offset in range(0, len(prompts), batch_size):
        chunk = prompts[offset : offset + batch_size]
        encoded = teacher.tok(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length + 8,
        ).to(device)
        with torch.inference_mode():
            logits = teacher.model(**encoded, logits_to_keep=1).logits[:, -1, :]
        for row in logits:
            results.append(
                [
                    torch.logsumexp(row[list(ids)], -1).item() if ids else -1e9
                    for ids in label_id_groups
                ]
            )
    return results
