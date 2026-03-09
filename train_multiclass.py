import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from preprocessing import preprocess
# Load split data
train_df = pd.read_csv("data/train_split.csv")
val_df = pd.read_csv("data/val_split.csv")
def load_images(dataframe):
    images, labels = [], []
    for _, row in dataframe.iterrows():
        img = cv2.imread(f"data/train_images/{row['id_code']}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess(img)
        images.append(img)
        labels.append(row['diagnosis'])
    return np.array(images), np.array(labels)
print("Loading images...")
X_train, y_train = load_images(train_df)
X_val, y_val = load_images(val_df)
y_train = to_categorical(y_train, 5)
y_val = to_categorical(y_val, 5)
# Model
base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base.output)
x = Dense(64, activation="relu")(x)
output = Dense(5, activation="softmax")(x)
model = Model(inputs=base.input, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=16
)
# Evaluation
y_pred = model.predict(X_val)
y_pred_class = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)
print(classification_report(y_true, y_pred_class))
cm = confusion_matrix(y_true, y_pred_class)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
plt.plot(history.history['accuracy'], label="train")
plt.plot(history.history['val_accuracy'], label="val")
plt.legend()
plt.title("Accuracy")
plt.show()
plt.plot(history.history['loss'], label="train")
plt.plot(history.history['val_loss'], label="val")
plt.legend()
plt.title("Loss")
plt.show()
model.save("dr_multiclass.keras")