import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from sklearn.model_selection import train_test_split


df = pd.read_json("archive (1)/Sarcasm_Headlines_Dataset_v2.json", lines=True)

vectorizer = keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=100)
wordList = []
for x in df["headline"]:
    y = x.split(" ")
    for word in y:
        wordList.append(word)


vectorizer.adapt(wordList)

sequences = vectorizer(df["headline"].to_list())
sequences = sequences.numpy()
labels = df["is_sarcastic"]
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, shuffle=True, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Input(shape=(100,)),
    keras.layers.Embedding(10000, 16),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation = "relu"),
    keras.layers.Dense(16, activation = "relu"),
    keras.layers.Dense(1, activation = "sigmoid")
])
optimizer = keras.optimizers.Adam(learning_rate = 0.005)
model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics =["accuracy"])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=30)


text1 = tf.constant(["Typhoon Rips Through Cemetery; Hundreds Dead"])
print(model.predict(vectorizer(text1).numpy()))


