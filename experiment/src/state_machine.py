
from enum import Enum
import random

import json


with open("/home/andyrajaobelina/work/42/PYTHON/experiment/input/function_calling_tests.json", "r", encoding="utf-8") as file:
    vocab = json.load(file)

copy_vocab = vocab


prompt_list = []
for entry in vocab:
    prompt_list.append([entry['prompt']])


name_list = [
            '"fn_add_numbers"',
            '"fn_greet"',
            '"fn_reverse_string"',
            '"fn_get_square_root"',
        ]

print(prompt_list)



class State(Enum):
    START = 1
    AFTER_OPEN_BRACE = 2
    AFTER_PROMPT_KEY = 3
    AFTER_PROMPT_COLON = 4
    AFTER_PROMPT_VALUE = 5
    AFTER_PROMPT_COMMA = 6
    AFTER_NAME_KEY = 7
    AFTER_NAME_COLON = 8
    AFTER_NAME_VALUE = 9
    AFTER_NAME_COMMA = 10
    AFTER_PARAMETERS_KEY = 11
    AFTER_PARAMETERS_COLON = 12
    AFTER_PARAMETERS_OPEN_BRACE = 13

    PARAMETER_KEY = 14
    PARAMETER_COLON = 15
    PARAMETER_VALUE = 16

    AFTER_PARAMETER_COMMA = 17

    AFTER_PARAMETERS_CLOSE_BRACE = 18

    AFTER_MAIN_CLOSE_BRACE = 19


def get_allowed_tokens(state, token_prompt, token_name):
    if state == State.START:
        return ["{"]

    if state == State.AFTER_OPEN_BRACE:
        return ['"prompt"']

    if state ==  State.AFTER_PROMPT_KEY:
        return [":"]

    if state == State.AFTER_PROMPT_COLON:
        return [random.choice(token_prompt)]

    if state ==  State.AFTER_PROMPT_VALUE:
        return [","]

    if state == State.AFTER_PROMPT_COMMA:
        return ['"name"']

    if state == State.AFTER_NAME_KEY:
        return [":"]

    if state == State.AFTER_NAME_COLON:
        return [random.choice(token_name)]

    if state == State.AFTER_NAME_VALUE:
        return [","]

    return []

def next_state(
    state,
    token
):
    if (
        state == State.START
        and token == "{"
    ):
        return State.AFTER_OPEN_BRACE

    if (
        state == State.AFTER_OPEN_BRACE
        and token == '"prompt"'
    ):
        return State.AFTER_PROMPT_KEY

    if (
        state == State.AFTER_PROMPT_KEY
        and token == ":"
    ):
        return State.AFTER_PROMPT_COLON

    if (
        state == State.AFTER_PROMPT_COLON
    ):
        return State.AFTER_PROMPT_VALUE

    if (
        state == State.AFTER_PROMPT_VALUE
        and token == ","
    ):
        return State.AFTER_PROMPT_COMMA

    if (
        state == State.AFTER_PROMPT_COMMA
        and token == '"name"'
    ):
        return State.AFTER_NAME_KEY

    if (
        state == State.AFTER_NAME_KEY
        and token == ":"
    ):
        return State.AFTER_NAME_COLON

    if (
        state == State.AFTER_NAME_COLON
    ):
        return State.AFTER_NAME_VALUE

    if (
        state == State.AFTER_NAME_VALUE
    ):
        return State.PARAMETER_KEY
    return 0


s = State.START

result = []
i = 0

while i < 10:
    token = "".join(get_allowed_tokens(s, prompt_list, name_list))
    state = next_state(s, token)
    result.append(token)
    result.append(" ")
    s = state
    i += 1


#print("".join(result))

