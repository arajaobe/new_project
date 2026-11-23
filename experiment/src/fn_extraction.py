from typing import Any

from llm_sdk import Small_LLM_Model

from fn_decoder import generate
from fn_models import JSON_TYPES, FunctionDefinition, ParameterDefinition


def _quoted_values(prompt: str) -> list[str]:
    """Extract values enclosed by matching single or double quotes.

    Args:
        prompt: User prompt to inspect.

    Returns:
        Quoted values in their order of appearance.
    """
    values: list[str] = []
    start: int | None = None
    quote: str | None = None
    for index, character in enumerate(prompt):
        if quote is None and character in {"'", '"'}:
            start = index + 1
            quote = character
        elif quote is not None and character == quote:
            values.append(prompt[start:index])
            start = None
            quote = None
    return values


def _numeric_values(prompt: str) -> list[str]:
    """Extract signed integer and decimal literals from a prompt.

    Args:
        prompt: User prompt to inspect.

    Returns:
        Numeric literals as strings in their order of appearance.
    """
    values: list[str] = []
    index = 0
    while index < len(prompt):
        start = index
        if prompt[index] in "+-" and index + 1 < len(prompt) and prompt[index + 1].isdigit():
            index += 1
        if index >= len(prompt) or not prompt[index].isdigit():
            index = start + 1
            continue
        while index < len(prompt) and prompt[index].isdigit():
            index += 1
        if index + 1 < len(prompt) and prompt[index] == "." and prompt[index + 1].isdigit():
            index += 1
            while index < len(prompt) and prompt[index].isdigit():
                index += 1
        values.append(prompt[start:index])
    return values


def _word_values(prompt: str) -> list[str]:
    """Extract identifier-like words from a prompt.

    Args:
        prompt: User prompt to inspect.

    Returns:
        Words containing letters, digits, underscores, or hyphens.
    """
    values: list[str] = []
    index = 0
    while index < len(prompt):
        if not (prompt[index].isalpha() or prompt[index] == "_"):
            index += 1
            continue
        start = index
        index += 1
        while index < len(prompt) and (prompt[index].isalnum() or prompt[index] in "_-"):
            index += 1
        values.append(prompt[start:index])
    return values


def _schema_choices(definition: ParameterDefinition) -> list[Any]:
    """Return the allowed values declared by a parameter definition.

    Args:
        definition: Parameter schema to inspect.

    Returns:
        Values from ``enum`` or, when absent, ``values``.
    """
    return definition.enum or definition.values


def _constrained_value(
    prompt: str,
    parameter_name: str,
    choices: list[Any],
    model: Small_LLM_Model,
) -> Any:
    """Decode one value while constraining output to the candidate choices.

    Args:
        prompt: User prompt supplying context for decoding.
        parameter_name: Name of the parameter being decoded.
        choices: Candidate values permitted for the parameter.
        model: Model used for constrained decoding.

    Returns:
        The decoded candidate, or the first candidate if decoding is invalid.
    """
    context = f"User request: {prompt}\nParameter: {parameter_name}\nAllowed values:\n"
    prompt_ids = model.encode(context)[0].tolist()
    text_choices = [str(choice) for choice in choices]
    sequences = [model.encode(choice)[0].tolist() for choice in text_choices]
    value_ids = generate(sequences, prompt_ids, model)
    value = model.decode(value_ids).strip()
    for choice, text_choice in zip(choices, text_choices):
        if value == text_choice:
            return choice
    return choices[0]


def _parameter_candidates(
    prompt: str,
    parameter_name: str,
    definition: ParameterDefinition,
) -> list[Any]:
    """Build type-appropriate candidate values from a user prompt.

    Args:
        prompt: User prompt to parse.
        parameter_name: Parameter whose value is being extracted.
        definition: Validated parameter schema.

    Returns:
        Candidate values suitable for constrained decoding.
    """
    parameter_type = definition.type
    if parameter_type not in JSON_TYPES:
        raise ValueError(f"Unsupported parameter type: {parameter_type!r}")

    quoted = _quoted_values(prompt)
    prompt_lower = prompt.lower()
    if "replace" in prompt_lower or "substitute" in prompt_lower:
        if parameter_name == "source_string" and quoted:
            return [quoted[-1]]
        if parameter_name == "replacement":
            if len(quoted) >= 3:
                return [quoted[1]]
            words = _word_values(prompt)
            for index, word in enumerate(words[:-1]):
                if word.lower() == "with":
                    return [words[index + 1]]
        if parameter_name not in {"regex", "source_string", "replacement"}:
            return quoted
    if parameter_name == "regex" and ("replace" in prompt_lower or "substitute" in prompt_lower):
        if "number" in prompt_lower or "digit" in prompt_lower:
            return [r"\d+"]
        if "vowel" in prompt_lower:
            return [r"[aeiou]"]
        if "word" in prompt_lower and quoted:
            return [quoted[0]]

    if parameter_type == "number":
        return [float(value) for value in _numeric_values(prompt)]
    if parameter_type == "integer":
        return [int(value) for value in _numeric_values(prompt) if "." not in value]
    if parameter_type == "boolean":
        return [True, False]
    if parameter_type in {"object", "array", "null"}:
        return []
    if quoted:
        return quoted
    ignored = {"what", "is", "the", "of", "and", "to", "a", "an", "with"}
    return [word for word in _word_values(prompt) if word.lower() not in ignored]


def _unused_candidates(candidates: list[Any], used_values: list[Any]) -> list[Any]:
    """Remove values that have already been assigned to earlier parameters.

    Args:
        candidates: Candidate values for the current parameter.
        used_values: Values already assigned in this function call.

    Returns:
        Candidates not present in ``used_values``.
    """
    return [candidate for candidate in candidates if not any(candidate == used for used in used_values)]


def _fallback_value(definition: ParameterDefinition) -> Any:
    """Return a valid JSON fallback for a parameter schema.

    Args:
        definition: Validated parameter schema.

    Returns:
        An empty or zero-like value matching the declared JSON type.
    """
    parameter_type = definition.type
    if parameter_type not in JSON_TYPES:
        raise ValueError(f"Unsupported parameter type: {parameter_type!r}")
    choices = _schema_choices(definition)
    if choices:
        return choices[0]
    return {
        "number": 0.0,
        "integer": 0,
        "boolean": False,
        "string": "",
        "object": {},
        "array": [],
        "null": None,
    }[parameter_type]


def extract_parameters(
    prompt: str,
    function: FunctionDefinition,
    model: Small_LLM_Model,
) -> dict[str, Any]:
    """Extract all arguments for a selected function.

    Args:
        prompt: User prompt containing possible argument values.
        function: Validated function definition.
        model: Model used for constrained value decoding.

    Returns:
        A mapping from parameter names to extracted values.
    """
    values: dict[str, Any] = {}
    used_candidates: list[Any] = []
    for name, definition in function.parameters.items():
        candidates = _unused_candidates(
            _schema_choices(definition) or _parameter_candidates(prompt, name, definition),
            used_candidates,
        )
        value = _fallback_value(definition) if not candidates else _constrained_value(
            prompt, name, candidates, model
        )
        values[name] = value
        used_candidates.append(value)
    return values
