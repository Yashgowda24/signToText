import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import pad_sequences

DATA_DIR = "processed_data"

data = []
labels = []
label_map = {}
class_index = 0

for file in os.listdir(DATA_DIR):
    label_name = file.replace(".npy", "")
    
    # You can split by _ if naming like 'hello_1.npy', 'thanks_2.npy'
    label_name = label_name.split("_")[0]

    if label_name not in label_map.values():
        label_map[class_index] = label_name
        class_index += 1

    label_id = list(label_map.keys())[list(label_map.values()).index(label_name)]
    
    data.append(np.load(os.path.join(DATA_DIR, file)))
    labels.append(label_id)

data = pad_sequences(data, padding='post', dtype='float32')
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Update the model architecture to handle the larger input size (126 features instead of 63)
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(data.shape[1], data.shape[2])),
    LSTM(128),
    Dense(128, activation='relu'),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")

# Save label_map for later prediction
import json
with open("models/label_map.json", "w") as f:
    json.dump(label_map, f)

print("Training complete. Model and label map saved.")
