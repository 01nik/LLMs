from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text= """Hello","I am learning LLMS","Will you help me to understand the every domain of it."""

tokens = tokenizer(text)
print("Input IDs:", tokens["input_ids"])
print("Attention Mask:", tokens["attention_mask"])
decoded=tokenizer.convert_ids_to_tokens(tokens["input_ids"])
print("Tokens:", decoded)