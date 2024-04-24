import random
import json
import pickle

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


# Download NLTK's components
nltk.download('punkt')
nltk.download('wordnet')

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data
with open("intents.json", "r") as file:
    intents = json.load(file)

# To store symptoms that user might say
words = []
# To store the disease they might be having
classes = []
# To store patten with its respective tags
documents = []

ignore_letters = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize each intents into words and classes that breaks longer
        # symptoms into single words
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))

        # Add intent tag to classes list if not already present
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize words and remove ignored characters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# To dump words & classes and to convert them into pickle file, which help you to convert
# Python object into a B(byte)/stream and save them in file or database for later usage
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)


# Prepare training data to put into nueral network
training_data = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training_data.append([bag, output_row])

# Shuffle and convert training data to numpy array
random.shuffle(training_data)
training_data = np.array(training_data, dtype=object)

# Split training data into input and output
train_x = list(training_data[:, 0])
train_y = list(training_data[:, 1])

# Build neural network model
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save model
model.save('chatbot_model.h5')
print("Done!")
