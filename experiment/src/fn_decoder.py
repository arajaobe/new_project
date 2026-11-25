import numpy as np

from llm_sdk import Small_LLM_Model


def generate(
    valid_sequences: list[list[int]],
    prompt_ids: list[int],
    model: Small_LLM_Model,
) -> list[int]:
    """Generate the most likely token sequence from a constrained set."""
    generated: list[int] = []
    candidates = valid_sequences

    while candidates:
        allowed_ids = {
            sequence[len(generated)]
            for sequence in candidates
            if len(sequence) > len(generated)
        }
        if not allowed_ids:
            break

        logits = model.get_logits_from_input_ids(prompt_ids + generated)
        masked_logits = np.full(len(logits), -np.inf)
        for token_id in allowed_ids:
            masked_logits[token_id] = logits[token_id]

        next_token = int(np.argmax(masked_logits))
        generated.append(next_token)
        candidates = [
            sequence
            for sequence in candidates
            if sequence[: len(generated)] == generated
        ]

    return generated
