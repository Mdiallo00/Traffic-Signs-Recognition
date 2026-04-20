import pandas as pd
import cv2
import numpy as np
import os

IMG_SIZE = 32

train_df = pd.read_csv("Train.csv")

X = []
y = []

for i in range(len(train_df)):
    img_path = os.path.join(train_df.loc[i, "Path"])
    label = train_df.loc[i, "ClassId"]

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    X.append(img)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)