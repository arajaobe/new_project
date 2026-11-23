
import numpy as np
import json

from llm_sdk import Small_LLM_Model

model = Small_LLM_Model(device="cpu")

with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as file:
    vocab = json.load(file)

reverse_vocab = {value: key for key, value in vocab.items()}

token_ids = model.encode("2 + 2 =")[0].tolist()

logits = model.get_logits_from_input_ids(token_ids)

best_token = np.argsort(logits)[-10:][::-1]

for token_id in best_token:
    print(token_id, reverse_vocab.get(token_id))

print("Score: ")
for score in best_token:
    print(logits[score])

temp = 0

while temp != 13:

    logits = model.get_logits_from_input_ids(
        token_ids
    )

    best_token = np.argmax(logits)

    temp = int(best_token)

    token_ids.append(best_token)

print(
    model.decode(token_ids)
)

#top_token = np.argmax(logits)

#print("logits: ", best_token)


#print(reverse_vocab[top_token])