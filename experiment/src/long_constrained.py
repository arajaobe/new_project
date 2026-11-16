
import numpy as np
import json

from llm_sdk import Small_LLM_Model

model = Small_LLM_Model(device="cpu")

with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as file:
    vocab = json.load(file)

reverse_vocab = {value: key for key, value in vocab.items()}

prompt = "the sky is"

token_ids = model.encode(prompt)[0].tolist()

logit = model.get_logit_by_ids(token_ids)

allowed = [
    ["Ġcovered", "Ġby", "Ġcloud"],
    ["Ġnot", "Ġnice"],
    ["Ġblue", "Ġtoday"]
    ]

masked_logit = logit




#while i < len(allowed):
#    j = 0
#    while j < len(allowed[i]):
#        print(allowed[i][j])
#        j +=1
#    i += 1

def find_position(matrix, target):
    for i, row in enumerate(matrix):
        if target in row:
            return [i, row.index(target)]
    return None  # Return None if not found

#print(find_position(allowed, 6419))
#print(find_position(allowed, 9437))
def check_next(matrix, pos):
    ind1 = int(pos[0])
    ind2 = int(pos[1])
    ko = 1
    try:
        matrix[ind1][ind2 + 1]
    except Exception:
        ko = 0
    return ko

firt_pos = []

for first in allowed:
    firt_pos.append(first[0])


#for i in range(len(masked_logits)):
#    if i not in allowed_ids:
#        masked_logits[i] = float("-inf")



print(firt_pos)




#res = check_next(allowed, 0)

#print(res)

#print(len(allowed[0]))

#print(reverse_vocab[20804])
#print(vocab["Ġblue"])