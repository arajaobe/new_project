
import numpy as np
import json
from pathlib import Path

from llm_sdk import Small_LLM_Model

model = Small_LLM_Model(device="cpu")



ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT_DIR / "data" / "input"

FUNCTIONS_FILE = (
    INPUT_DIR /
    "functions_definition.json"
)

TESTS_FILE = (
    INPUT_DIR /
    "function_calling_tests.json"
)


with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as file:
    vocab = json.load(file)

reverse_vocab = {value: key for key, value in vocab.items()}

with open(FUNCTIONS_FILE, "r", encoding="utf-8") as file:
    vocab_func = json.load(file)

with open(TESTS_FILE, "r", encoding="utf-8") as file:
    vocab_prompt = json.load(file)


prompt_list = []
for entry in vocab_prompt:
    prompt_list.append(entry['prompt'])


func_name = []
for entry in vocab_func:
    func_name.append([entry["name"], entry["description"]])


context = "Available functions:\n\n"

for function in func_name:

    context += (
        f"Function: {function[0]}\n"
        f"Description: {function[1]}\n\n"
    )

context += (
    f"User request:\n{prompt_list[0]}\n\n"
    "Selected function:\n"
)



#with open("your_file.txt", "w", encoding="utf-8") as file_write:
#    for name, description in func_name:
#        file_write.write(f"function: {name}\n")
#        file_write.write(f"description: {description}\n")



#prompt = "fn_get_square_root. Calculate the square root of a number."

#prompt =  "What is the square root of 16. fn_get_square_root. fn_reverse_string. fn_add_number"


#prompt = f"""
#Available functions:

#Function: fn_add_numbers
#Description: Add two numbers together and return their sum.

#Function: fn_greet
#Description: Generate a greeting message for a person.

#Function: fn_reverse_string
#Description: Reverse a string.

#Function: fn_get_square_root
#Description: Calculate the square root of a number.

#Function: fn_substitute_string_with_regex
#Description: Replace all occurrences matching a regex pattern in a string.

#User request:
#Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'

#Selected function:
#"""

#prompt = "12.32"

prompt = context


token_ids = model.encode(prompt)[0].tolist()

logits = model.get_logits_from_input_ids(token_ids)






text1 = "covered by cloud"
text2 = "not nice"
text3 = "blue today"


#valid_sequence = [
#    [20804, 553, 9437],
#    [1921, 6419],
#    [12203, 3351]
#    ]


valid_sequence = [
    [8822, 1889, 3744],
    [8822, 2891, 32964],
    [8822, 43277, 3904],
    [8822, 3062, 39794, 12993],
    [8822, 5228, 7660, 3904, 6615, 41832]
    ]



#result_seq = []

#valid_sequence = [
#    ["Ġcovered", "Ġby", "Ġcloud"],
#    ["Ġnot", "Ġnice"],
#    ["Ġblue", "Ġtoday"]
#    ]

#valid_sequence = [
#    ["Ġa", "Ġbeautiful", "Ġplace"],
#    ["Ġa", "Ġbeautiful", "Ġplanet"],
#    ["Ġa", "Ġbeautiful", "Ġworld"]
#    ]

#valid_sequence = [
#    ["fn", "_add", "_number"],
#    ["fn", "_g", "reet"],
#    ["fn", "_reverse", "_string"],
#    ["fn", "_get", "_square", "_root"]
#    ]






def generating(valid_sequence, prompt_ids):

    j = 3
    current_prefix = []

    while j:
        allowed_pos = []


        len_pos = len(current_prefix)
        for pos in valid_sequence:
            if len(pos) > len_pos:
                allowed_pos.append(pos[len_pos])



        if not allowed_pos:
            break

        #allowed_ids = {token for token in allowed_pos}
        allowed_ids = set(allowed_pos)


        masked_logits = logits.copy()

        for i in range(len(masked_logits)):
            if i not in allowed_ids:
                masked_logits[i] = float("-inf")


        top_token = np.argmax(masked_logits)

        prompt_ids.append(top_token)


        current_prefix.append(top_token)

        #print("chosen:", model.decode(current_prefix))

        matching_sequences = []

        for seq in valid_sequence:

            if seq[:len(current_prefix)] == current_prefix:
                matching_sequences.append(seq)

        valid_sequence = matching_sequences

    return current_prefix


result_prompt = generating(valid_sequence, token_ids)

actual_prompt = "".join(vocab_prompt[0]["prompt"])


decoded_prompt = model.decode(result_prompt)
#print('"prompt":', actual_prompt)
print(f'"name": {decoded_prompt}')

#print(model.decode(result_prompt))


description = []
for entry in vocab_func:
    if entry["name"] == decoded_prompt:
        description.append(entry["description"])

description_text = "".join(description)

#print(description)
parameters = []

for entry in vocab_func:
    if entry["name"] == decoded_prompt and entry["description"] == description_text:
        parameters.append(entry["parameters"])


res = [entry for entry in parameters]
keys = []

output_param = ""
for inner_list in res:

    my_dict = inner_list


    for key, info in my_dict.items():
        output_param += f"{key} ({info['type']})\n"
        keys.append([key])


print(keys)


#context_param = (
#"Function:\n"
#f"{decoded_prompt}\n\n"
#"Description:\n"
#f'{description_text}\n\n'
#"Expected parameters:\n"
#f"{output_param}\n\n"
#"User request:\n"
#f"{prompt_list[0]}\n\n"
#"Parameter name:\n"
#)
#print(context_param)


context_param = (
"Function:\n"
f"{decoded_prompt}\n\n"
"Description:\n"
f'{description_text}\n\n'
"Expected parameters:\n"
f"{output_param}\n\n"
"User request:\n"
f"{prompt_list[0]}\n\n"
"Value:\n"
)

#new_valiud_sequence = [
#    [64],
#    [65]
#]

#new_valiud_sequence = [
#    [18457],
#    [2005],
#    [47817],
#    [43111]
#    ]

new_valiud_sequence = [
    [17],
    [18]
    ]




new_prompt = context_param


new_token_ids = model.encode(new_prompt)[0].tolist()

new_logits = model.get_logits_from_input_ids(new_token_ids)


new_result = generating(new_valiud_sequence, new_token_ids)

print("parameter name:", model.decode(new_result))

print(model.decode(new_token_ids))

#print(parameters)




#print(context_param)