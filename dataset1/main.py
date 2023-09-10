import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import random

DATA_PATH = 'emotions.csv'
RANDOM_STATE = 123
TEST_SIZE = 0.3
LABEL_MAPPING = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
EMOTION_LABELS = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

data = pd.read_csv(DATA_PATH)
data['label'] = data['label'].map(LABEL_MAPPING)

emotions = data['label'].unique()
num_features = {emotion: {'significant': 0, 'non-significant': 0} for emotion in emotions}

for emotion in emotions:
    subset = data[data['label'] == emotion]
    for feature in data.columns[:-1]:
        _, p_value = ttest_ind(subset[feature], data[feature])
        if p_value < 0.05:
            num_features[emotion]['significant'] += 1
        else:
            num_features[emotion]['non-significant'] += 1

scaler = StandardScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.2, epochs=70, batch_size=32, verbose=2)

model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Test Accuracy: {model_acc * 100:.3f}%")

y_pred = np.argmax(model.predict(X_test), axis=-1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=LABEL_MAPPING.keys(), yticklabels=LABEL_MAPPING.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

clr = classification_report(y_test, y_pred, target_names=LABEL_MAPPING.keys())
print("Classification Report:\n", clr)

for i in range(6, 8):
    random_index = random.randint(0, len(X_test) - 1)
    sample_input = X_test.iloc[random_index].values.reshape(1, -1)
    true_label = y_test.iloc[random_index]
    true_emotion = EMOTION_LABELS[true_label]
    predicted_emotion = model.predict(sample_input)
    predicted_label = EMOTION_LABELS[np.argmax(predicted_emotion)]
    print(f"Sample {i}: Real Emotion Label: {true_emotion}, Predicted Emotion Label: {predicted_label}")

model.save('emotions_model.h5')
