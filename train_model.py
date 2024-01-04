# train_model.py
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from preprocessing import preprocess_data
from model_definition import create_lstm_model
import pandas as pd

# Load the training and validation datasets
train_data = pd.read_csv('twitter_training.csv')
train_data['tweet'].fillna('', inplace=True)

val_data = pd.read_csv('twitter_validation.csv')
val_data['tweet'].fillna('', inplace=True)

# Assuming you have train_data with 'tweet' and 'sentiment' columns

# Preprocess the data
max_length = 256  # Adjust based on dataset and requirements
train_padded, tokenizer = preprocess_data(train_data['tweet'], max_length)

# Encode the labels
num_classes = len(train_data['sentiment'].unique())
train_labels = to_categorical(train_data['sentiment'].astype('category').cat.codes)

# Split the data
train_padded, val_padded, train_labels, val_labels = train_test_split(train_padded, train_labels, test_size=0.2, random_state=42)

# Define the model
vocab_size = len(tokenizer.word_index) + 1  # Add 1 for the OOV token
embedding_dim = 256
lstm_units = 256
model = create_lstm_model(vocab_size, embedding_dim, max_length, lstm_units, num_classes)

# Train the model
model.fit(train_padded, train_labels, epochs=5, validation_data=(val_padded, val_labels))

# Save the model
model.save('sentiment_analysis_model.h5')

# Save the tokenizer 
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer_json)