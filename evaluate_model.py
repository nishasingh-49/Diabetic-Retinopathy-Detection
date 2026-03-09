import pandas as pd
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from preprocessing import preprocess
# Load saved model
model = load_model("dr_multiclass.keras")
print("\nModel Summary:")
model.summary()
# Load validation data
val_df = pd.read_csv("data/val_split.csv")
def load_images(dataframe):
    images, labels = [], []
    for i, row in dataframe.iterrows():
        img = cv2.imread(f"data/train_images/{row['id_code']}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess(img)
        images.append(img)
        labels.append(row['diagnosis'])
    return np.array(images), np.array(labels)
print("\nLoading validation images...")
X_val, y_val = load_images(val_df)
y_true = y_val
# Predictions
y_pred = model.predict(X_val)
y_pred_class = np.argmax(y_pred, axis=1)
# Metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred_class))
# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_class)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()