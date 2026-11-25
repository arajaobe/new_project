import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from llm_sdk import Small_LLM_Model

from .fn_function_call import build_function_call
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root_dir = Path.cwd()
    parser = argparse.ArgumentParser(description="Run function-calling prompts.")
    parser.add_argument(
        "--functions_definition",
        "--functions-definition",
        type=Path,
        default=root_dir / "data" / "input" / "functions_definition.json",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root_dir / "data" / "input" / "function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root_dir / "data" / "output" / "function_calling_results.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model = Small_LLM_Model(device="cpu")
    with args.input.open(encoding="utf-8") as file:
        prompts = json.load(file)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        file.write("[\n")
        for index, prompt in enumerate(prompts):
            result = build_function_call(
                prompt["prompt"],
                functions_definition=args.functions_definition,
                model=model,
            )
            serialized = json.dumps(result, indent=2)
            if index:
                file.write(",\n")
            file.write(serialized)
            file.flush()
            print(serialized, flush=True)
        file.write("\n]\n")


if __name__ == "__main__":
    main()
