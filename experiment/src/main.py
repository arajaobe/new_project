import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from llm_sdk import Small_LLM_Model

from .fn_function_call import build_function_call
from .fn_models import load_function_definitions


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for running function-calling prompts.

    Args:
        argv: Optional command-line arguments; uses the process arguments
            when omitted.

    Returns:
        Parsed command-line options, including the input and output paths.
    """
    root_dir = Path.cwd()
    parser = argparse.ArgumentParser(
        description="Run function-calling prompts."
    )
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
    """Run the prompt inputs and write their function-call results.

    Args:
        argv: Optional command-line arguments; uses the process arguments
            when omitted.

    Returns:
        None.
    """
    args = parse_args(argv)
    try:
        functions = load_function_definitions(args.functions_definition)
    except json.JSONDecodeError as error:
        print(
            "Invalid JSON in function definitions file "
            f"'{args.functions_definition}' "
            f"at line {error.lineno}, column {error.colno}: {error.msg}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    except OSError as error:
        print(
            f"Could not read function definitions file "
            f"'{args.functions_definition}': {error}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    except ValidationError as error:
        first = error.errors()[0]
        location = ".".join(
            map(str, first["loc"])
        )
        print(
            f"Invalid function definition: "
            f"{location}: {first['msg']}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    try:
        with args.input.open(encoding="utf-8") as file:
            prompts = json.load(file)
    except json.JSONDecodeError as error:
        print(
            f"Invalid JSON in prompt input file '{args.input}' "
            f"at line {error.lineno}, column {error.colno}: {error.msg}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    except OSError as error:
        print(
            f"Could not read prompt input file '{args.input}': {error}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    model = Small_LLM_Model(device="cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        file.write("[\n")
        for index, prompt in enumerate(prompts):
            try:
                result = build_function_call(
                    prompt["prompt"],
                    functions=functions,
                    model=model,
                )
                result = {"prompt": prompt["prompt"], **result}
                serialized = json.dumps(result, indent=2)
            except Exception as error:
                print(
                    f"Error processing prompt {index+1}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                result = {
                    "prompt": prompt["prompt"],
                    "name": "",
                    "parameters": {}
                }
                serialized = json.dumps(result, indent=2)
            if index:
                file.write(",\n")
            file.write(serialized)
            file.flush()
            print(f"Processed prompt {index+1}/{len(prompts)}", flush=True)
            print(serialized, flush=True)
        file.write("\n]\n")


if __name__ == "__main__":
    main()
