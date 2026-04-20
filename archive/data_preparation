import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

IMG_SIZE = 32

train_df = pd.read_csv("Train.csv")

X = []
y = []

for i in range(len(train_df)):
    img_path = train_df.loc[i, "Path"]
    label = train_df.loc[i, "ClassId"]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    X.append(img)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of classes:", len(np.unique(y)))

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)

plt.figure(figsize=(8, 8))
for i in range(9):
    idx = random.randint(0, len(X_train) - 1)
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[idx])
    plt.title(f"Class {y_train[idx]}")
    plt.axis("off")

plt.tight_layout()
plt.show()