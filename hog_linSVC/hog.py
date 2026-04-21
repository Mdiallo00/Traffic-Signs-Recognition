import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================
# CONFIG
# =========================
IMG_SIZE = (32, 32)
RANDOM_STATE = 42

TRAIN_CSV_PATH = os.path.join("archive", "Train.csv")
TEST_CSV_PATH = os.path.join("archive", "Test.csv")
BASE_DIR = "archive"

print("===== Loading Report =====")

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

X_train = []
y_train = []
train_skipped = 0

print("\nExtracting HOG features from training set...")
for i in range(len(train_df)):
    relative_path = train_df.loc[i, "Path"]
    label = train_df.loc[i, "ClassId"]
    img_path = os.path.join(BASE_DIR, relative_path)

    if not os.path.exists(img_path):
        train_skipped += 1
        continue

    img = Image.open(img_path).convert("L")
    img = img.resize(IMG_SIZE)
    img_np = np.array(img)

    features = hog(
        img_np,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

    X_train.append(features)
    y_train.append(label)

    if (i + 1) % 5000 == 0 or (i + 1) == len(train_df):
        print(f"Processed {i + 1}/{len(train_df)} training images")

X_test = []
y_test = []
test_skipped = 0

print("\nExtracting HOG features from test set...")
for i in range(len(test_df)):
    relative_path = test_df.loc[i, "Path"]
    label = test_df.loc[i, "ClassId"]
    img_path = os.path.join(BASE_DIR, relative_path)

    if not os.path.exists(img_path):
        test_skipped += 1
        continue

    img = Image.open(img_path).convert("L")
    img = img.resize(IMG_SIZE)
    img_np = np.array(img)

    features = hog(
        img_np,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

    X_test.append(features)
    y_test.append(label)

    if (i + 1) % 5000 == 0 or (i + 1) == len(test_df):
        print(f"Processed {i + 1}/{len(test_df)} test images")

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test)

print("\nDataset loaded successfully")
print(f"Total training images: {len(X_train)}")
print(f"Skipped training images: {train_skipped}")
print(f"Total test images: {len(X_test)}")
print(f"Skipped test images: {test_skipped}")

# =========================
# MODEL
# =========================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LinearSVC(
        C=0.1,
        max_iter=3000,
        tol=1e-3,
        random_state=RANDOM_STATE
    ))
])

# =========================
# TRAIN
# =========================
start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()

# =========================
# PREDICT
# =========================
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)

y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print("\n===== Summary =====")
results_df = pd.DataFrame({
    "Train": [train_acc],
    "Test": [test_acc]
}, index=["HOG + LinearSVC"])

print(results_df.to_string())

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("===== Matrix =====")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(14, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", values_format="d")
plt.title("HOG + LinearSVC Confusion Matrix")
plt.tight_layout()
plt.savefig("hog_linearsvc_confusion_matrix.png", bbox_inches="tight")
plt.show()