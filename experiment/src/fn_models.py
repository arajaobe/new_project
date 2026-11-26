import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


JSON_TYPES = {
    "string",
    "number",
    "integer",
    "boolean",
    "object",
    "array",
    "null",
}
JsonType = Literal[
    "string",
    "number",
    "integer",
    "boolean",
    "object",
    "array",
    "null",
]


class ParameterDefinition(BaseModel):
    """Schema for a single function parameter.

    Attributes:
        type: JSON type accepted for the parameter.
        enum: Optional enumerated choices.
        values: Optional alternate list of allowed values.
    """

    model_config = ConfigDict(extra="ignore")

    type: JsonType
    enum: list[Any] = Field(default_factory=list)
    values: list[Any] = Field(default_factory=list)


class FunctionDefinition(BaseModel):
    """Schema describing a callable function and its parameters.

    Attributes:
        name: Function name.
        description: Human-readable purpose of the function.
        parameters: Parameter schemas keyed by parameter name.
        returns: Optional schema for the function's return value.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class FunctionCall(BaseModel):
    """A selected function name together with its parameter values.

    Attributes:
        name: Name of the selected function.
        parameters: Values supplied for the function's parameters.
    """

    name: str
    parameters: dict[str, Any]


def load_function_definitions(
    functions_definition: str | Path,
) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        functions_definition: Path to the JSON file containing definitions.

    Returns:
        Validated function definitions from the file.
    """
    with Path(functions_definition).open(encoding="utf-8") as file:
        return [
            FunctionDefinition.model_validate(item)
            for item in json.load(file)
        ]
