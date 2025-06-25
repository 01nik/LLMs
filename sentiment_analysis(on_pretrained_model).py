from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")
#Default model used when not specified--> (distilbert-base-uncased-finetuned-sst-2-english)

texts = ["The movie was not as bad as I expected, but I wouldn't watch it again.",
    "This product is terrible. Total waste of money.",
    "It's okay, not great but not bad either."]

for text in texts:
    result = sentiment_pipeline(text)[0]
    print(f"Text: {text}")
    print(f"Label: {result['label']}, Score: {round(result['score'], 3)}\n")