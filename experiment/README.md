*This project has been created as part of the 42 curriculum by arajaobe.*

# Call Me Maybe

## Description

Call Me Maybe is a function-calling system built around the provided Qwen/Qwen3-0.6B model and constrained decoding.

The goal is to transform natural-language requests into structured function calls while guaranteeing valid JSON output and schema compliance.

## Installation

### Standard installation

```bash
make install
```

or

```bash
uv sync
```

### Installation in Goinfre

For 42 environments with limited home-directory storage:

```bash
export UV_CACHE_DIR="/goinfre/arajaobe/.cache/uv"
export HF_HOME="/goinfre/arajaobe/.huggingface"
export TMPDIR="/goinfre/arajaobe/tmp"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TMPDIR"
```

Then install normally:

```bash
uv sync
```

This stores:

- uv cache in Goinfre
- HuggingFace model cache in Goinfre
- temporary files in Goinfre

## Usage

```bash
uv run python -m src
```

Custom paths:

```bash
uv run python -m src     --functions_definition data/input/functions_definition.json     --input data/input/function_calling_tests.json     --output data/output/function_calling_results.json
```

## Algorithm Explanation

The project implements token-level constrained decoding.

1. Build a context from function descriptions and the user prompt.
2. Tokenize all valid function names.
3. Generate one token at a time.
4. Obtain model logits.
5. Keep only logits corresponding to valid continuations.
6. Set all invalid logits to negative infinity.
7. Select the highest-scoring valid token.
8. Repeat until a complete valid sequence is generated.
9. Extract and validate parameters.
10. Produce valid JSON output.

## Design Decisions

- Pydantic is used for schema validation.
- Function names are selected using constrained decoding.
- Parameter values are selected from schema-compliant candidates.
- Function descriptions are included in model context.
- Fallback values guarantee schema-valid JSON.
- The model is initialized once and reused.

## Performance Analysis

- Single model load.
- Deterministic argmax decoding.
- Valid JSON generation.
- Schema validation for every function call.
- Designed to process the provided prompt set within the project limits.

## Challenges Faced

- Understanding tokenizer behaviour.
- Decoding multi-token function names.
- Building prefix-based constrained decoding.
- Ensuring schema compliance.
- Handling malformed input files gracefully.

## Testing Strategy

The implementation was tested with:

- arithmetic prompts
- string manipulation prompts
- regex replacement prompts
- malformed JSON files
- missing input files
- enum-restricted values
- boolean parameters
- integer parameters
- decimal values

## Example

Input:

```text
What is the sum of 2 and 3?
```

Output:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

## Resources

- Python documentation
- Pydantic documentation
- NumPy documentation
- Qwen/Qwen3-0.6B documentation

## AI Usage

AI was used for architecture discussions, constrained decoding design, schema validation discussions, testing ideas, debugging guidance, and documentation drafting. All generated suggestions were reviewed, tested, and understood before being incorporated.
