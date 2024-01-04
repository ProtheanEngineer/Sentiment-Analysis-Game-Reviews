# test_model.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
loaded_model = load_model('sentiment_analysis_model.h5')

# Load the tokenizer
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    loaded_tokenizer_json = f.read()
    loaded_tokenizer = tokenizer_from_json(loaded_tokenizer_json)

# Example test text
test_texts = ["Must say that Elden Ring is the best game ever.", 
              "This game is awful. It has a lot of bugs everywhere.", 
              "When is the new Mass Effect coming?"]

# Preprocess the test data
max_length = 256  # Adjust based on dataset and requirements
test_sequences = loaded_tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Make predictions
predictions = loaded_model.predict(test_padded)

# Convert predictions to sentiment labels
sentiment_labels = ['Positive', 'Negative', 'Neutral', 'Irrelevant']

for i, prediction in enumerate(predictions):
    predicted_sentiment = sentiment_labels[prediction.argmax()]
    print(prediction)
    print(f"Text: {test_texts[i]}, Predicted Sentiment: {predicted_sentiment}")
