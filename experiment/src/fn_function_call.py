import json
from pathlib import Path
from typing import Any

from llm_sdk import Small_LLM_Model

from .fn_decoder import generate
from .fn_extraction import extract_parameters
from .fn_models import (
    FunctionCall,
    FunctionDefinition,
    load_function_definitions,
)


def select_function(
    prompt: str,
    functions: list[FunctionDefinition],
    model: Small_LLM_Model,
) -> FunctionDefinition:
    context = (
        "Available functions:\n\n"
        + "\n".join(
            f"Function: {function.name}\nDescription: {function.description}\n"
            for function in functions
        )
        + f"\nUser request: {prompt}\n\nSelected function:\n"
    )
    prompt_ids = model.encode(context)[0].tolist()
    sequences = [model.encode(function.name)[0].tolist() for function in functions]
    selected_name = model.decode(generate(sequences, prompt_ids, model)).strip()
    return next((function for function in functions if function.name == selected_name), functions[0])


def _validate_call(call: FunctionCall, function: FunctionDefinition) -> None:
    if set(call.parameters) != set(function.parameters):
        raise ValueError(f"Invalid parameter keys for {function.name!r}")
    for name, definition in function.parameters.items():
        value = call.parameters[name]
        valid = {
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "string": isinstance(value, str),
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "null": value is None,
        }[definition.type]
        if not valid:
            raise ValueError(f"Invalid {definition.type} value for parameter {name!r}")
        choices = definition.enum or definition.values
        if choices and value not in choices:
            raise ValueError(f"Invalid enum value for parameter {name!r}")


def build_function_call(
    prompt: str,
    *,
    functions: list[FunctionDefinition | dict[str, Any]] | None = None,
    functions_definition: str | Path | None = None,
    model: Small_LLM_Model | None = None,
) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if functions is None and functions_definition is None:
        raise ValueError("functions or functions_definition must be provided")
    definitions = [
        item if isinstance(item, FunctionDefinition) else FunctionDefinition.model_validate(item)
        for item in (
            functions
            if functions is not None
            else load_function_definitions(functions_definition)
        )
    ]
    inference_model = model or Small_LLM_Model(device="cpu")
    function = select_function(prompt, definitions, inference_model)
    call = FunctionCall(
        name=function.name,
        parameters=extract_parameters(prompt, function, inference_model),
    )
    _validate_call(call, function)
    return call.model_dump()
