
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model(device="cpu")

print(model.get_path_to_vocab_file())

