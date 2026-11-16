
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model(device="cpu")

text1 = "covered by cloud"
text2 = " fn"
text3 = "blue today"
text4 = "fn_greet"
text5 = "add"
text6 = "2"
text7 = "sum"
text8 = "3"



#allowed = [
#    [20804, 553, 9437],
#    [1921, 6419],
#    [12203, 3351]
#    ]


#allowed = [
#    [18457],
#    [2005],
#    [47817],
#    [43111]
#    ]

allowed = [
    [718],
    [20],
    [1242],
    [4082]
    ]


#allowed = [
#    [8822, 1889, 3744],
#    [ 8822,  2891, 32964],
#    [ 8822, 43277,  3904],
#    [ 8822,  3062, 39794, 12993],
#    [ 8822,  5228,  7660,  3904,  6615, 41832]
#    ]


new_allowed = [
    ['What is the sum of 2 and 3?'],
    ['What is the sum of 265 and 345?'],
    ['Greet shrek'],
    ['Greet john'],
    ["Reverse the string 'hello'"],
    ["Reverse the string 'world'"],
    ['What is the square root of 16?'],
    ['Calculate the square root of 144'],
    ['Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS'],
    ["Replace all vowels in 'Programming is fun' with asterisks"],
    ["Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"]]


#new_allowed = [
#    [3838,  374,  279, 2629,  315,  220,   17,  323,  220,   18,   30],
#    [3838,  374,  279, 2629,  315,  220,   17,   21,   20,  323,  220, 18,   19,   20,   30],
#    [38,  3744,   557, 41861],
#    [38,  3744, 39642],
#    [45695,   279,   914,   364, 14990,     6],
#    [45695,   279,   914,   364, 14615,     6],
#    [3838,  374,  279, 9334, 3704,  315,  220,   16,   21,   30],
#    [47866,   279,  9334,  3704,   315,   220,  16, 19,  19],
#    [23107,   678,  5109,   304,   330,  9707, 220, 18, 19, 358, 2776, 220, 17, 18, 18, 1635, 2310, 1, 448, 15943, 44116],
#    [23107, 678, 78456, 304, 364, 45719, 374, 2464,  6,  448, 33937, 92834],
#    [3136, 7660, 279, 3409, 364, 4616, 6, 448, 364, 18457, 6, 304, 364, 785, 8251, 7578, 389, 279, 517, 448, 2441, 8251, 6]]



#for item in new_allowed:
#    print(model.encode(item))



t = allowed[0]
current_prefix = 20804


matching_sequences = []

#for seq in allowed:

#    if seq[:len(current_prefix)] == current_prefix:
#        matching_sequences.append(seq)

match_seq = [20804, 553, 9437]

allwd = match_seq
test_seq = [1, 2, 3]

#for x in allowed:
#    print(len(x))
if matching_sequences:
    print("ok")
else:
    print("ko")

#for seq in allowed:
#    print(seq[:1])

#print(t[:2])

tokens1 = model.encode(text1)
tokens2 = model.encode(text2)

tokens4 = model.encode(text4)
tokens5 = model.encode(text5)
tokens6 = model.encode(text6)
tokens7 = model.encode(text7)
tokens8 = model.encode(text8)


#for t in allowed:
#    print(model.decode(t))

#decode1 = model.decode(3744)


#decoded = model.decode(tokens1[0])

#res =int(tokens[0])

#print("Original text:", text)

#print("\nTokens:", tokens1)
#print("\nTokens:", tokens2)
#print("\nTokens:", tokens4)
print("\nTokens:", tokens5)
print("\nTokens:", tokens6)
print("\nTokens:", tokens7)
print("\nTokens:", tokens8)





#print("\nType: ", type(tokens))

#print("\nType: ", type(res))
#print(res)


#print("\nDecoded: ", decode1)

