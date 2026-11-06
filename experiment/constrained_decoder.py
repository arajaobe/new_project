
import numpy as np
import json

from llm_sdk import Small_LLM_Model

model = Small_LLM_Model(device="cpu")

with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as file:
    vocab = json.load(file)

reverse_vocab = {value: key for key, value in vocab.items()}


allowed_ids = {
    vocab["Ġblue"],
    vocab["Ġclear"],
    vocab["Ġcloudy"],
}

token_ids = model.encode("The sky is")[0].tolist()

logits = model.get_logits_from_input_ids(token_ids)

masked_logits = logits.copy()

for i in range(len(masked_logits)):
    if i not in allowed_ids:
        masked_logits[i] = float("-inf")



best_token = np.argsort(masked_logits)[-3:][::-1]

for result_token in best_token:
    print(result_token, reverse_vocab.get(result_token))


top_token = np.argmax(masked_logits)

print(f"The sky is{model.decode(top_token)}")

for token_id in best_token:
    print(
        token_id,
        masked_logits[token_id],
        repr(reverse_vocab[token_id])
    )
