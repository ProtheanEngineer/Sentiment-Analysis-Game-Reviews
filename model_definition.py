# model_definition.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_lstm_model(vocab_size, embedding_dim, max_length, lstm_units, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        Bidirectional(LSTM(256 , return_sequences=True)) , 
        Bidirectional(LSTM(128)) , 
        Dense(64 , activation='relu') , 
        Dropout(0.5) , 
        Dense(4 , activation='softmax')
        # LSTM(units=lstm_units),
        # Dense(units=num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
