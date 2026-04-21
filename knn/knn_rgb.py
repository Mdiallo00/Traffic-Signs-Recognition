import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

IMG_SIZE = 32
N_NEIGHBORS = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42

CSV_PATH = os.path.join("archive", "Train.csv")
BASE_DIR = "archive"

print("===== Loading Report =====")

train_df = pd.read_csv(CSV_PATH)

X = []
y = []
skipped = 0

for i in range(len(train_df)):
    relative_path = train_df.loc[i, "Path"]
    label = train_df.loc[i, "ClassId"]
    img_path = os.path.join(BASE_DIR, relative_path)

    img = cv2.imread(img_path)
    if img is None:
        skipped += 1
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    X.append(img)
    y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)

knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
knn.fit(X_train_flat, y_train)

train_pred = knn.predict(X_train_flat)
train_acc = accuracy_score(y_train, train_pred)

y_pred = knn.predict(X_val_flat)
val_acc = accuracy_score(y_val, y_pred)

print(f"Dataset loaded successfully")
print(f"Total images: {len(X)}")
print(f"Skipped images: {skipped}")

print("\n===== Summary =====")
results_df = pd.DataFrame({
    "Train": [train_acc],
    "Val": [val_acc]
}, index=["k-NN RGB"])

print(results_df.to_string())
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))

print("===== Matrix =====")
cm = confusion_matrix(y_val, y_pred)

fig, ax = plt.subplots(figsize=(14, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", values_format="d")
plt.title("k-NN RGB Confusion Matrix")
plt.tight_layout()
plt.savefig("knn_rgb_confusion_matrix.png", bbox_inches="tight")
plt.show()