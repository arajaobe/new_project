
import json

with open("/home/andyrajaobelina/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json","r", encoding="utf-8") as file:
    vocab=json.load(file)



print(type(vocab))
print(len(vocab))

for key, value in list(vocab.items())[:20]:
    print(repr(key), value)

reverse_vocab = {value: key for key, value in vocab.items()}


#print(reverse_vocab[13])
print("ok" in vocab)
print("Ġ," in vocab)
