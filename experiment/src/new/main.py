"""Command-line entry point for streaming function-call results."""

import json

from llm_sdk import Small_LLM_Model

from fn_function_call import build_function_call
from fn_models import OUTPUT_FILE, TESTS_FILE


def main() -> None:
    """Generate and stream function calls for all configured test prompts.

    Results are printed as they are generated and written incrementally to
    ``data/output/function_calling_results.json``.
    """
    model = Small_LLM_Model(device="cpu")
    with TESTS_FILE.open(encoding="utf-8") as file:
        prompts = json.load(file)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        file.write("[\n")
        for index, prompt in enumerate(prompts):
            result = build_function_call(prompt["prompt"], model=model)
            serialized = json.dumps(result, indent=2)
            if index:
                file.write(",\n")
            file.write(serialized)
            file.flush()
            print(serialized, flush=True)
        file.write("\n]\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")