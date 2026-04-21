import os
import time
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# -----------------------------
# CONFIG
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 43
batch_size = 32

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = base_dir
test_csv_path = os.path.join(root_dir, "Test.csv")
model_path = os.path.join(root_dir, "resnet18_gtsrb_transfer.pth")

print("Current device:", device)
print("Root directory:", root_dir)
print("Model path:", model_path)

# -----------------------------
# PATH CHECKS
# -----------------------------
if not os.path.exists(test_csv_path):
    raise FileNotFoundError(f"Could not find Test.csv: {test_csv_path}")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Could not find saved model: {model_path}")

# -----------------------------
# TRANSFORMS
# -----------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# CUSTOM DATASET
# -----------------------------
class GTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        required_columns = ["Path", "ClassId"]
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(
                    f"Column '{col}' not found in {csv_file}. "
                    f"Columns found: {list(self.data.columns)}"
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_relative_path = str(self.data.iloc[idx]["Path"]).strip()
        label = int(self.data.iloc[idx]["ClassId"])

        img_relative_path = img_relative_path.replace("\\", os.sep).replace("/", os.sep)

        if img_relative_path.startswith("." + os.sep):
            img_relative_path = img_relative_path[2:]

        img_path = os.path.join(self.root_dir, img_relative_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"\nImage not found.\n"
                f"Row index: {idx}\n"
                f"CSV path: {img_relative_path}\n"
                f"Full path tried: {img_path}\n"
            )

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# -----------------------------
# LOAD TEST DATASET
# -----------------------------
test_dataset = GTSRBDataset(
    csv_file=test_csv_path,
    root_dir=root_dir,
    transform=test_transform
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Total test images:", len(test_dataset))

# -----------------------------
# LOAD MODEL
# -----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

print("Model loaded successfully.")

# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_with_details(model, loader):
    model.eval()
    all_labels = []
    all_preds = []

    prediction_start_time = time.time()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    prediction_end_time = time.time()
    prediction_time_seconds = prediction_end_time - prediction_start_time

    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted"
    )

    return accuracy, cm, report, precision, recall, f1, prediction_time_seconds

accuracy, cm, report, precision, recall, f1_score, prediction_time = evaluate_with_details(model, test_loader)

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\n========== EVALUATION RESULTS ==========")
print(f"Accuracy        : {accuracy:.4f}")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"F1-Score        : {f1_score:.4f}")
print(f"Prediction Time : {prediction_time:.2f} seconds")

print("\nClassification Report:")
print(report)

# -----------------------------
# SAVE CLASSIFICATION REPORT
# -----------------------------
report_path = os.path.join(root_dir, "classification_report_resnet18.txt")
with open(report_path, "w") as f:
    f.write("========== EVALUATION RESULTS ==========\n")
    f.write(f"Accuracy        : {accuracy:.4f}\n")
    f.write(f"Precision       : {precision:.4f}\n")
    f.write(f"Recall          : {recall:.4f}\n")
    f.write(f"F1-Score        : {f1_score:.4f}\n")
    f.write(f"Prediction Time : {prediction_time:.2f} seconds\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"\nClassification report saved at: {report_path}")

# -----------------------------
# PLOT + SAVE CONFUSION MATRIX
# -----------------------------
cm_path = os.path.join(root_dir, "confusion_matrix_resnet18.png")

plt.figure(figsize=(16, 12))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - ResNet18 on GTSRB")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(cm_path)
plt.show()

print(f"Confusion matrix saved at: {cm_path}")