import re

text = "I bought 2 apples for 2.3 dollars each, and 10 bananas. Now I have -52.5 dollars left."

# Regex pattern to match integers and decimals
pattern = r"-?\d+(?:\.\d+)?"

# Find all matches in the text (they are returned as strings)
matches = re.findall(pattern, text)

print(matches)
# Output: ['2', '2.3', '10']



# Convert each matched string into a float or int
numbers = [float(n) if '.' in n else int(n) for n in matches]

print(numbers)
# Output: [2, 2.3, 10]