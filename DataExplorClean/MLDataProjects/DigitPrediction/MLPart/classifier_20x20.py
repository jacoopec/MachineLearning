import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

# Save path for images
SAVE_DIR = "dummy_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Generate and save dummy 20x20 images
def generate_dummy_data(num_samples=1000, save_images=True):
    X = torch.randn(num_samples, 1, 20, 20)  # [B, C, H, W]
    y = torch.randint(0, 10, (num_samples,))

    if save_images:
        for i in range(num_samples):
            img_array = X[i].squeeze().numpy()
            plt.imsave(f"{SAVE_DIR}/img_{i}_label_{y[i].item()}.png", img_array, cmap="gray")

    return TensorDataset(X, y)

# Simple CNN for classification
class CNN20x20(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN20x20, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Training setup
batch_size = 32
epochs = 5
learning_rate = 0.001

dataset = generate_dummy_data(num_samples=100, save_images=True)  # Save 100 example images
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN20x20().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

print(f"✅ Training done. Images saved to ./{SAVE_DIR}")
