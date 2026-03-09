import pandas as pd
import cv2
import matplotlib.pyplot as plt
from preprocessing import preprocess

# Load CSV
df = pd.read_csv("data/train.csv")

# Show first 3 images
for i in range(3):
    img_id = df.iloc[i]['id_code']
    label = df.iloc[i]['diagnosis']

    # Load image
    img = cv2.imread(f"data/train_images/{img_id}.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess image
    img_clean = preprocess(img)

    # Plot raw vs cleaned
    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title(f"Raw | Label: {label}")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img_clean)
    plt.title("Preprocessed")
    plt.axis("off")

    plt.show()