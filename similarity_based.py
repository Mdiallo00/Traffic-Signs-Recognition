import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score

# settings
img_size = 32

# load dataset
train_data = pd.read_csv("Train.csv")

images = []
labels = []

# process images
for i in range(len(train_data)):
    path = train_data.loc[i, "Path"]
    label = train_data.loc[i, "ClassId"]

    img = cv2.imread(path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0

    images.append(img)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

# flatten images
X = images.reshape(len(images), -1)
y = labels

print("Flattened shape:", X.shape)

# split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# KNN
print("\nTraining KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_preds = knn.predict(X_val)
knn_acc = accuracy_score(y_val, knn_preds)
print("KNN Accuracy:", round(knn_acc, 4))

# Radius Neighbors
print("\nTraining Radius Neighbors...")
radius_model = RadiusNeighborsClassifier(radius=3.0, outlier_label=-1)
radius_model.fit(X_train, y_train)

radius_preds = radius_model.predict(X_val)

valid_idx = radius_preds != -1

if np.sum(valid_idx) > 0:
    radius_acc = accuracy_score(y_val[valid_idx], radius_preds[valid_idx])
    print("Radius Accuracy:", round(radius_acc, 4))
else:
    print("Radius model could not classify samples.")