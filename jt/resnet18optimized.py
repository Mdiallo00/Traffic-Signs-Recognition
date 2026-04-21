import os
import copy
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================
# CONFIG
# =========================
NUM_CLASSES = 43
BATCH_SIZE = 32
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 5
RANDOM_STATE = 42

STAGE1_LR = 5e-4
STAGE2_LR = 5e-5

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "archive"))
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "Train.csv")
TEST_CSV_PATH = os.path.join(BASE_DIR, "Test.csv")

# =========================
# DEVICE PRIORITY
# =========================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("===== Loading Report =====")
print("Current device:", DEVICE)

# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(12),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.08, 0.08),
        scale=(0.90, 1.10)
    ),
    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASET
# =========================
class GTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_rel_path = self.data.iloc[idx]["Path"]
        label = int(self.data.iloc[idx]["ClassId"])

        img_path = os.path.join(self.root_dir, img_rel_path)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

print(f"Training rows: {len(train_df)}")
print(f"Test rows: {len(test_df)}")

train_dataset = GTSRBDataset(TRAIN_CSV_PATH, BASE_DIR, transform=train_transform)
train_eval_dataset = GTSRBDataset(TRAIN_CSV_PATH, BASE_DIR, transform=eval_transform)
test_dataset = GTSRBDataset(TEST_CSV_PATH, BASE_DIR, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_eval_loader = DataLoader(train_eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Dataset loaded successfully")
print(f"Total training images: {len(train_dataset)}")
print(f"Total test images: {len(test_dataset)}")

# =========================
# MODEL
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=STAGE1_LR)

# =========================
# EVALUATION FUNCTION
# =========================
def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

# =========================
# TRAIN FUNCTION
# =========================
def train_model(model, train_loader, train_eval_loader, test_loader, criterion, optimizer, epochs):
    best_weights = copy.deepcopy(model.state_dict())
    best_test_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        y_train_epoch, train_pred_epoch = get_predictions(model, train_eval_loader)
        train_acc_epoch = accuracy_score(y_train_epoch, train_pred_epoch)

        y_test_epoch, test_pred_epoch = get_predictions(model, test_loader)
        test_acc_epoch = accuracy_score(y_test_epoch, test_pred_epoch)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Train Acc: {train_acc_epoch:.4f} | "
            f"Test Acc: {test_acc_epoch:.4f}"
        )

        if test_acc_epoch > best_test_acc:
            best_test_acc = test_acc_epoch
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model

# =========================
# STAGE 1
# =========================
print("\nStage 1: Training final classifier only")
model = train_model(
    model,
    train_loader,
    train_eval_loader,
    test_loader,
    criterion,
    optimizer,
    EPOCHS_STAGE1
)

# =========================
# STAGE 2
# =========================
print("\nStage 2: Fine-tuning layer4 and fc")

for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=STAGE2_LR
)

model = train_model(
    model,
    train_loader,
    train_eval_loader,
    test_loader,
    criterion,
    optimizer,
    EPOCHS_STAGE2
)

# =========================
# FINAL PREDICTIONS
# =========================
y_train, train_pred = get_predictions(model, train_eval_loader)
train_acc = accuracy_score(y_train, train_pred)

y_test, y_pred = get_predictions(model, test_loader)
test_acc = accuracy_score(y_test, y_pred)

# =========================
# SUMMARY
# =========================
print("\n===== Summary =====")
results_df = pd.DataFrame({
    "Train": [train_acc],
    "Test": [test_acc]
}, index=["ResNet18"])

print(results_df.to_string())

# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# =========================
# CONFUSION MATRIX
# =========================
print("===== Matrix =====")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(14, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", values_format="d")
plt.title("ResNet18 Confusion Matrix")
plt.tight_layout()
plt.savefig("resnet18optimized_confusion_matrix.png", bbox_inches="tight")
plt.show()

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "resnet18optimized_gtsrb_transfer.pth")
print("Model saved as: resnet18optimized_gtsrb_transfer.pth")