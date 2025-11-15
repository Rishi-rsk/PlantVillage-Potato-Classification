import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import get_model
from tqdm import tqdm

data_dir = "data/train"
val_dir = "data/val"

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_ds = datasets.ImageFolder(data_dir, transform=transform)
val_ds   = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model(3).to(device)

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.0001)

epochs = 10

for epoch in range(epochs):
    print(f"\nEPOCH {epoch+1}/{epochs}")

    # Training
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        opt.zero_grad()
        preds = model(imgs)
        loss = loss_fn(preds, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            _, predicted = preds.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Train Loss: {total_loss / len(train_loader):.4f}")
    print(f"Val Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "saved_model/potato_model.pth")
print("Model saved!")
