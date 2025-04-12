# Byte Pair Encoding (BPE)

Dataset: 
- GLUE (General Language Understanding Evaluation)
- CodeSearchNet (Python subset of CodeSearchNet)

Framework: 
- Torch


## Steps

1. Preprocess Input Text:
    - Convert text into a sequence of characters and count the frequency of character pairs.
    - Use a dictionary to store word frequencies.

2. Merge Rules:
    - Identify the most frequent pair of characters and merge them into a new token.
    - Update the word frequencies after each merge.

3. Vocabulary Construction:
    - Repeat the merge process until the desired vocabulary size is reached.
    - Save the merge rules and vocabulary.

4. Tokenization:
    - Apply the learned merge rules to tokenize new input text.
