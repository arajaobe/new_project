
from llm_sdk import Small_LLM_Model

import numpy as np

model = Small_LLM_Model(device="cpu")

token_ids = model.encode(
    "The Big Bang Theory"
)[0].tolist()

temp = 0

while temp != 13:

    logits = model.get_logits_from_input_ids(
        token_ids
    )

    best_token = np.argmax(logits)

    temp = int(best_token)

    token_ids.append(best_token)

print(
    model.decode(20804)
)