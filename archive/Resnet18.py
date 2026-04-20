import os
import copy
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

# -----------------------------
# CONFIG
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 43
batch_size = 32
epochs_stage1 = 5
epochs_stage2 = 5

# Build paths relative to this Python file


# base_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.join(base_dir, "archive")

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = base_dir
train_csv_path = os.path.join(root_dir, "Train.csv")
test_csv_path = os.path.join(root_dir, "Test.csv")

print("Current device:", device)
print("Base directory:", base_dir)
print("Root directory:", root_dir)

# Quick path checks
if not os.path.exists(root_dir):
    raise FileNotFoundError(f"Could not find dataset folder: {root_dir}")

if not os.path.exists(train_csv_path):
    raise FileNotFoundError(f"Could not find Train.csv: {train_csv_path}")

if not os.path.exists(test_csv_path):
    raise FileNotFoundError(f"Could not find Test.csv: {test_csv_path}")

print("Files inside archive folder:")
print(os.listdir(root_dir))

# -----------------------------
# TRANSFORMS
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
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

        # Make sure required columns exist
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
        img_relative_path = self.data.iloc[idx]["Path"]
        label = int(self.data.iloc[idx]["ClassId"])

        img_path = os.path.join(self.root_dir, img_relative_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# -----------------------------
# LOAD FULL DATASETS
# -----------------------------
full_train_dataset = GTSRBDataset(
    csv_file=train_csv_path,
    root_dir=root_dir,
    transform=train_transform
)

test_dataset = GTSRBDataset(
    csv_file=test_csv_path,
    root_dir=root_dir,
    transform=val_test_transform
)

print("Total training images:", len(full_train_dataset))
print("Total test images:", len(test_dataset))

# -----------------------------
# SPLIT TRAIN INTO TRAIN + VALIDATION
# -----------------------------
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=generator
)

# Important note:
# random_split creates subsets that still point to the same dataset object.
# So changing dataset.transform affects both subsets.
# To avoid that issue cleanly, we rebuild separate datasets below.

train_indices = train_dataset.indices
val_indices = val_dataset.indices

full_train_data = pd.read_csv(train_csv_path)

train_df = full_train_data.iloc[train_indices].reset_index(drop=True)
val_df = full_train_data.iloc[val_indices].reset_index(drop=True)

train_split_csv = os.path.join(root_dir, "train_split_temp.csv")
val_split_csv = os.path.join(root_dir, "val_split_temp.csv")

train_df.to_csv(train_split_csv, index=False)
val_df.to_csv(val_split_csv, index=False)

train_dataset = GTSRBDataset(
    csv_file=train_split_csv,
    root_dir=root_dir,
    transform=train_transform
)

val_dataset = GTSRBDataset(
    csv_file=val_split_csv,
    root_dir=root_dir,
    transform=val_test_transform
)

print("Training split size:", len(train_dataset))
print("Validation split size:", len(val_dataset))

# -----------------------------
# DATALOADERS
# -----------------------------
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# LOAD PRETRAINED RESNET18
# -----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all pretrained layers first
for param in model.parameters():
    param.requires_grad = False

# Replace final fully connected layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# -----------------------------
# LOSS FUNCTION + OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_weights = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # ---- Training ----
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * inputs.size(0)
            train_correct += torch.sum(preds == labels).item()
            train_total += labels.size(0)

        train_loss = train_loss_total / train_total
        train_acc = train_correct / train_total

        # ---- Validation ----
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                val_loss_total += loss.item() * inputs.size(0)
                val_correct += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        val_loss = val_loss_total / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model, history

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    return correct / total

# -----------------------------
# STAGE 1: TRAIN FINAL CLASSIFIER ONLY
# -----------------------------
print("\nStage 1: Training final classifier only")

model, history_stage1 = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs_stage1
)

# -----------------------------
# STAGE 2: FINE-TUNE LAYER4 + FC
# -----------------------------
print("\nStage 2: Fine-tuning layer4 and fc")

for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

model, history_stage2 = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs_stage2
)

# -----------------------------
# TEST EVALUATION
# -----------------------------
test_acc = evaluate(model, test_loader)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
model_path = os.path.join(base_dir, "resnet18_gtsrb_transfer.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved successfully at: {model_path}")