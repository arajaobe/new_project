
import numpy as np
import json

from llm_sdk import Small_LLM_Model

model = Small_LLM_Model(device="cpu")

with open("/home/andyrajaobelina/work/42/PYTHON/experiment/input/functions_definition.json", "r", encoding="utf-8") as file:
    vocab = json.load(file)

#for key, value in list(vocab.items()):
#    print(repr(key), value)

print(type(vocab))

for i in vocab:
    print(repr(i))