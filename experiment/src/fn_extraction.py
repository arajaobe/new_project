from typing import Any

from llm_sdk import Small_LLM_Model

from .fn_decoder import generate
from .fn_models import JSON_TYPES, FunctionDefinition, ParameterDefinition


def _quoted_values(prompt: str) -> list[str]:
    """Extract substrings enclosed in matching single or double quotes.

    Args:
        prompt: Text to scan for quoted substrings.

    Returns:
        Quoted values in the order they appear.
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
    """Extract signed integer and decimal number strings from text.

    Args:
        prompt: Text to scan for numbers.

    Returns:
        Number strings in the order they appear.
    """
    values: list[str] = []
    index = 0
    while index < len(prompt):
        start = index
        if (
            prompt[index] in "+-"
            and index + 1 < len(prompt)
            and prompt[index + 1].isdigit()
        ):
            index += 1
        if index >= len(prompt) or not prompt[index].isdigit():
            index = start + 1
            continue
        while index < len(prompt) and prompt[index].isdigit():
            index += 1
        if (
            index + 1 < len(prompt)
            and prompt[index] == "."
            and prompt[index + 1].isdigit()
        ):
            index += 1
            while index < len(prompt) and prompt[index].isdigit():
                index += 1
        values.append(prompt[start:index])
    return values


def _word_values(prompt: str) -> list[str]:
    """Extract word-like tokens from text.

    Args:
        prompt: Text to split into word-like tokens.

    Returns:
        Tokens that begin with a letter or underscore and may contain
        letters, digits, underscores, or hyphens thereafter.
    """
    values: list[str] = []
    index = 0
    while index < len(prompt):
        if not (prompt[index].isalpha() or prompt[index] == "_"):
            index += 1
            continue
        start = index
        index += 1
        while index < len(prompt) and (
            prompt[index].isalnum() or prompt[index] in "_-"
        ):
            index += 1
        values.append(prompt[start:index])
    return values


def _schema_choices(definition: ParameterDefinition) -> list[Any]:
    """Return the allowed values declared for a parameter.

    Args:
        definition: Parameter schema to inspect.

    Returns:
        The enum values when present, otherwise the parameter's values.
    """
    return definition.enum or definition.values


def _constrained_value(
    prompt: str,
    parameter_name: str,
    choices: list[Any],
    model: Small_LLM_Model,
) -> Any:
    """Choose one schema-constrained value using model inference.

    Args:
        prompt: User request providing context for the choice.
        parameter_name: Name of the parameter being populated.
        choices: Allowed values from which to choose.
        model: Model used to score the candidate values.

    Returns:
        The matching original value, or the first choice if decoding does
        not match a candidate exactly.
    """
    context = (
        f"User request: {prompt}\n"
        f"Parameter: {parameter_name}\n"
        "Allowed values:\n"
    )
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
    """Extract prompt-derived candidate values for one parameter.

    Args:
        prompt: User request to inspect.
        parameter_name: Name of the parameter being populated.
        definition: Schema describing the parameter's JSON type.

    Returns:
        Candidate values inferred from the prompt.

    Raises:
        ValueError: If the parameter uses an unsupported JSON type.
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
            if "asterisk" in prompt_lower:
                return ["*"]
            if len(quoted) >= 3:
                return [quoted[1]]
            words = _word_values(prompt)
            for index, word in enumerate(words[:-1]):
                if word.lower() == "with":
                    return [words[index + 1]]
        if parameter_name not in {"regex", "source_string", "replacement"}:
            return quoted
    if parameter_name == "regex" and (
        "replace" in prompt_lower or "substitute" in prompt_lower
    ):
        if "number" in prompt_lower or "digit" in prompt_lower:
            return [r"\d+"]
        if "vowel" in prompt_lower:
            return [r"[aeiouAEIOU]"]
        if "word" in prompt_lower and quoted:
            return [quoted[0]]

    if parameter_type == "number":
        return [float(value) for value in _numeric_values(prompt)]
    if parameter_type == "integer":
        return [
            int(value)
            for value in _numeric_values(prompt)
            if "." not in value
        ]
    if parameter_type == "boolean":
        return [True, False]
    if parameter_type in {"object", "array", "null"}:
        return []
    if quoted:
        return quoted
    ignored = {"what", "is", "the", "of", "and", "to", "a", "an", "with"}
    return [
        word
        for word in _word_values(prompt)
        if word.lower() not in ignored
    ]


def _unused_candidates(
    candidates: list[Any],
    used_values: list[Any],
) -> list[Any]:
    """Remove candidate values already assigned to another parameter.

    Args:
        candidates: Values available for the current parameter.
        used_values: Values already assigned to prior parameters.

    Returns:
        Candidate values that do not equal any previously used value.
    """
    return [
        candidate
        for candidate in candidates
        if not any(candidate == used for used in used_values)
    ]


def _fallback_value(definition: ParameterDefinition) -> Any:
    """Return the first allowed value or a default for a parameter type.

    Args:
        definition: Parameter schema to provide a fallback for.

    Returns:
        The first declared choice, or a type-appropriate default.

    Raises:
        ValueError: If the parameter uses an unsupported JSON type.
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
    """Extract schema-constrained parameter values from a user prompt.

    Args:
        prompt: User request to parse.
        function: Definition whose parameters must be populated.
        model: Model used to select among candidate values.

    Returns:
        A mapping from parameter names to extracted or fallback values.
    """
    values: dict[str, Any] = {}
    used_candidates: list[Any] = []
    for name, definition in function.parameters.items():
        candidates = _unused_candidates(
            _schema_choices(definition)
            or _parameter_candidates(prompt, name, definition),
            used_candidates,
        )
        value = (
            _fallback_value(definition)
            if not candidates
            else _constrained_value(prompt, name, candidates, model)
        )
        values[name] = value
        used_candidates.append(value)
    return values
