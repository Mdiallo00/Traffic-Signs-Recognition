
# 🚦 Traffic Sign Classification - Full Pipeline (ResNet34 + Augmentation + Grad-CAM)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

# ========================
# 🔧 CONFIG
# ========================
TRAIN_DIR = "data/GTSRB/Train"
TEST_DIR = "data/GTSRB/Test"
NUM_CLASSES = 43
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# 🔄 DATA AUGMENTATION
# ========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========================
# 🧠 MODEL (ResNet34)
# ========================
def get_model():
    model = models.resnet34(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES)
    )

    return model

model = get_model().to(device)

# ========================
# ⚙️ TRAINING SETUP
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# ========================
# 🔁 TRAIN FUNCTION
# ========================
def train(model, loader):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ========================
# 🧪 EVALUATION FUNCTION
# ========================
def evaluate(model, loader):
    model.eval()
    preds, labels_list = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            labels_list.extend(labels.numpy())

    acc = accuracy_score(labels_list, preds)
    cm = confusion_matrix(labels_list, preds)

    return acc, cm

# ========================
# 🚀 TRAIN LOOP
# ========================
for epoch in range(EPOCHS):
    loss = train(model, train_loader)
    acc, _ = evaluate(model, test_loader)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# ========================
# 📊 CONFUSION MATRIX
# ========================
def plot_cm(cm):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

acc, cm = evaluate(model, test_loader)
print("Final Accuracy:", acc)
plot_cm(cm)

# ========================
# 🔥 GRAD-CAM
# ========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, class_idx=None):
        self.model.eval()

        output = self.model(input_image)

        if class_idx is None:
            class_idx = output.argmax()

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam / cam.max()

        return cam.detach().cpu().numpy()

# ========================
# 🎯 GRAD-CAM VISUALIZATION
# ========================
def show_gradcam(model, image, label):
    grad_cam = GradCAM(model, model.layer4[-1])

    input_tensor = image.unsqueeze(0).to(device)
    cam = grad_cam.generate(input_tensor)

    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    heatmap = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = heatmap / 255 + img
    overlay = overlay / overlay.max()

    plt.imshow(overlay)
    plt.title(f"Grad-CAM (Label: {label})")
    plt.axis("off")
    plt.show()

# ========================
# 🔍 RUN GRAD-CAM ON SAMPLE
# ========================
images, labels = next(iter(test_loader))
show_gradcam(model, images[0], labels[0].item())

