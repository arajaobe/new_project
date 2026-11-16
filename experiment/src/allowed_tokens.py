from enum import Enum
import random
from typing import Literal

import json
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model(device="cpu")

from click import prompt


with open("/home/andyrajaobelina/work/42/PYTHON/experiment/data/input/functions_definition.json", "r", encoding="utf-8") as file:
    vocab_func = json.load(file)

with open("/home/andyrajaobelina/work/42/PYTHON/experiment/data/input/function_calling_tests.json", "r", encoding="utf-8") as file:
    vocab_prompt = json.load(file)


prompt_list = []
for entry in vocab_prompt:
    prompt_list.append(entry['prompt'])


#func_name = []
#for entry in vocab_func:
#    #func_name.append([entry["name"], entry["description"]])
#    print(entry)

parameters = []
for entry in vocab_func:
    parameters.append([entry["parameters"]])


#i = 0
#while len(parameters) > i:
#    res = [entry for entry in parameters[i]]

#    for inner_list in res:

#        my_dict = inner_list

#        for key, info in my_dict.items():
#            print(f"Key: {key}, Type: {info['type']}")
#    i += 1
res = [entry for entry in parameters[0]]

for inner_list in res:

    my_dict = inner_list

    for key, info in my_dict.items():
        print(f"Key: {key}, Type: {info['type']}")

print(my_dict)


#i = 0
#while len(parameters) > i:
#    for entry in parameters[i]:
#        #print(model.encode("".join(entry)))
#        print(" ".join(entry))

#    i += 1

#data = {'a': {'type': 'number'}, 'b': {'type': 'number'}}

#print(parameters)

# Loop through key and value pairs
#for key, info in parameters.items():
#    print(f"Key: {key}, Type: {info['type']}")


#arguments = []
#for argument in parameters:
#    arguments.append(argument)



#with open("your_file.txt", "w", encoding="utf-8") as file_write:
#    for name, description in func_name:
#        file_write.write(f"function: {name}\n")
#        file_write.write(f"description: {description}\n")

#for name, description in func_name:
#    print("function:", name)
#    print("description:", description)
#with open("your_file.txt", "r", encoding="utf-8") as file_read:
    #for line in file_read:
    #    print(line.strip())


#print(func_name)