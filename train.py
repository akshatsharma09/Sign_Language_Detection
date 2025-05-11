import os
import numpy as np
from function import *
from sklearn.model_selection import train_test_split
from keras.src.utils import to_categorical
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.callbacks import TensorBoard

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")

            # Ensure file exists before loading
            if os.path.exists(file_path):
                res = np.load(file_path, allow_pickle=True)
                
                # Ensure res always has shape (63,)
                if res.shape != (63,):
                    res = res.flatten()[:63]  # Flatten if necessary, truncate if oversized
                    if res.shape[0] < 63:
                        res = np.pad(res, (0, 63 - res.shape[0]), mode='constant')  # Pad if too short
            else:
                print(f"File not found: {file_path}")
                res = np.zeros((63,))  # Placeholder for missing frames

            window.append(res)

        # Convert window to a uniform NumPy array using `np.vstack()` for stacking
        window_array = np.vstack(window)  # Shape will be (sequence_length, 63)
        sequences.append(window_array)
        labels.append(label_map[action])

# Convert sequences to a NumPy array
X = np.array(sequences, dtype=np.float32)

# Convert labels to categorical format
y = to_categorical(labels).astype(np.int32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Model Architecture
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))  # Fixed shape reference

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

# Save Model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('model.h5')