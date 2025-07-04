import os
import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from cnn_model import CNNClassifier
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Load Feedback Images
class FeedbackDataset(Dataset):
    def __init__(self, label_file):
        with open(label_file, "r") as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = Image.open(sample["image"])
        label = sample["label"]
        return self.transform(image), label

if os.path.exists("feedback_labels.json"):
    feedback_data = FeedbackDataset("feedback_labels.json")
    from torch.utils.data import ConcatDataset
    train_data = ConcatDataset([train_data, feedback_data])
else:
    print("⚠️ No feedback data found. Retraining on MNIST only.")

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Train
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"✅ Epoch {epoch+1}/5: Loss {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "cnn_digit.pth")
print("🎉 Retraining complete. Model saved.")
