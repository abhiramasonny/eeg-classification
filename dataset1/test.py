import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

DATA_PATH = 'emotions.csv'
LABEL_MAPPING = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
EMOTION_LABELS = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

data = pd.read_csv(DATA_PATH)
data['label'] = data['label'].map(LABEL_MAPPING)

scaler = StandardScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])
X = data.drop('label', axis=1)
y = data['label']

loaded_model = tf.keras.models.load_model('emotions_model.h5')
y_pred = np.argmax(loaded_model.predict(X), axis=-1)

cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=LABEL_MAPPING.keys(), yticklabels=LABEL_MAPPING.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

clr = classification_report(y, y_pred, target_names=LABEL_MAPPING.keys())
print("Classification Report:\n", clr)
