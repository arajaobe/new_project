import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


JSON_TYPES = {"string", "number", "integer", "boolean", "object", "array", "null"}
JsonType = Literal["string", "number", "integer", "boolean", "object", "array", "null"]


class ParameterDefinition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: JsonType
    enum: list[Any] = Field(default_factory=list)
    values: list[Any] = Field(default_factory=list)


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition | None = None


class FunctionCall(BaseModel):
    name: str
    parameters: dict[str, Any]


def load_function_definitions(functions_definition: str | Path) -> list[FunctionDefinition]:
    with Path(functions_definition).open(encoding="utf-8") as file:
        return [FunctionDefinition.model_validate(item) for item in json.load(file)]
