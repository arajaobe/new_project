
import numpy as np
import json

from llm_sdk import Small_LLM_Model

model = Small_LLM_Model(device="cpu")

with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as file:
    vocab = json.load(file)

reverse_vocab = {value: key for key, value in vocab.items()}

prompt = "the sky is"

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

result_seq = []

valid_sequence = [
    ["Ġcovered", "Ġby", "Ġcloud"],
    ["Ġnot", "Ġnice"],
    ["Ġblue", "Ġtoday"]
    ]


def generating(valid) :
    i = 0


    for pos in valid_sequence:
        if len(pos) > i:
            allowed_pos.append(pos[i])


allowed_pos = []


i = 0


for pos in valid_sequence:
    if len(pos) > i:
        allowed_pos.append(pos[i])


#print(allowed_pos)


allowed_ids = {vocab[token] for token in allowed_pos}



masked_logits = logits.copy()

for i in range(len(masked_logits)):
    if i not in allowed_ids:
        masked_logits[i] = float("-inf")


top_token = np.argmax(masked_logits)

current_prefix  = []
current_prefix.append(reverse_vocab[top_token])

#print(f"The sky is{model.decode(top_token)}")
#print(prefix)





matching_sequences = []

for seq in valid_sequence:

    if seq[:len(current_prefix)] == current_prefix:
        matching_sequences.append(seq)

print(matching_sequences)