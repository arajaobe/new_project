import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "data" / "input"
FUNCTIONS_FILE = INPUT_DIR / "functions_definition.json"
TESTS_FILE = INPUT_DIR / "function_calling_tests.json"
OUTPUT_FILE = ROOT_DIR / "data" / "output" / "function_calling_results.json"
JSON_TYPES = {"string", "number", "integer", "boolean", "object", "array", "null"}
JsonType = Literal["string", "number", "integer", "boolean", "object", "array", "null"]


class ParameterDefinition(BaseModel):
    """Describe a function parameter and its optional allowed values."""

    model_config = ConfigDict(extra="ignore")

    type: JsonType
    enum: list[Any] = Field(default_factory=list)
    values: list[Any] = Field(default_factory=list)


class FunctionDefinition(BaseModel):
    """Describe a callable function exposed to the language model."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition | None = None


class FunctionCall(BaseModel):
    """Represent a validated function name and its arguments."""

    name: str
    parameters: dict[str, Any]


def load_function_definitions() -> list[FunctionDefinition]:
    """Load and validate function definitions from the input JSON file.

    Returns:
        A list of validated function definitions.
    """
    with FUNCTIONS_FILE.open(encoding="utf-8") as file:
        return [FunctionDefinition.model_validate(item) for item in json.load(file)]
