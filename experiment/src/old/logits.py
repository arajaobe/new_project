import numpy as np

from llm_sdk import Small_LLM_Model



model = Small_LLM_Model(device="cpu")

text = "two + two is equal to"

tokens = model.encode(text)

token_ids = tokens[0].tolist()

logits = model.get_logits_from_input_ids(token_ids)

print("type: ", type(logits))

#print("length: ", len(logits))

#print("tokens in list: ", token_ids)

#print(token_ids[2])

#print(model.decode(token_ids[2]))

best_logits = np.argsort(logits)[-10:][::-1]

print("logits: ", best_logits)

for score in best_logits:
    print(logits[score])

best_token = np.argmax(logits)

result = model.decode([best_token])

print(text + " " + repr(result))