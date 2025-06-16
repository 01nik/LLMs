from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


sentences = [
    "I love AI",                            # 3 words
    "Transformers are amazing tools",       # 4 words
    "Hello"                                 # 1 word
]

tokens = tokenizer(sentences, padding=True, truncation=True)


print("Input IDs:")
for ids in tokens['input_ids']:
    print(ids)

print("\nAttention Masks:")
for mask in tokens['attention_mask']:
    print(mask)
