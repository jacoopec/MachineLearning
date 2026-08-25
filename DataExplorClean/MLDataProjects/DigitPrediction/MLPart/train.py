import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# Directories
data_dir = './data'  # Must contain 3 subfolders for each class
model_save_path = './model.pth'
scripted_model_path = './model_scripted.pt'

# Hyperparameters
num_classes = 3
epochs = 10
batch_size = 7
learning_rate = 0.001
image_size = (20, 20)

# Transforms
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(image_size),
    transforms.ToTensor()
])

# Dataset and loader
dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = dataset.classes
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Save model
torch.save(model.state_dict(), model_save_path)
print(f"✅ Model saved to {model_save_path}")

# Evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
print(f"✅ Accuracy: {acc*100:.2f}%")
print(f"✅ Confusion Matrix:\n{cm}")

# TorchScript export
example_input = torch.rand(1, 1, *image_size).to(device)
scripted_model = torch.jit.trace(model, example_input)
scripted_model.save(scripted_model_path)
print(f"✅ TorchScript model saved to {scripted_model_path}")
