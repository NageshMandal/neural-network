from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
from torchvision import transforms
import base64
import json
import os
import uuid
from cnn_model import CNNClassifier
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = "cnn_digit.pth"

model = CNNClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    img_data = data['image']
    image = Image.open(io.BytesIO(base64.b64decode(img_data.split(",")[1]))).convert("L")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return jsonify({"prediction": predicted.item()})

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    img_data = data["image"]
    label = int(data["actual"])

    # Create folder
    os.makedirs("feedback_data", exist_ok=True)

    # Save image file
    img_bytes = base64.b64decode(img_data.split(",")[1])
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join("feedback_data", filename)
    img.save(filepath)

    # Append to feedback_labels.json
    labels_file = "feedback_labels.json"
    label_entry = {"image": filepath, "label": label}
    if not os.path.exists(labels_file):
        with open(labels_file, "w") as f:
            json.dump([label_entry], f, indent=2)
    else:
        with open(labels_file, "r+") as f:
            existing = json.load(f)
            existing.append(label_entry)
            f.seek(0)
            json.dump(existing, f, indent=2)

    # Retrain model on MNIST + feedback
    retrain_model()

    return jsonify({"status": "✅ Feedback saved and model retrained."})

def retrain_model():
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    print("🔁 Retraining model on feedback...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_local = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST
    mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform_local)

    # Feedback Dataset
    class FeedbackDataset(Dataset):
        def __init__(self, label_file):
            with open(label_file, "r") as f:
                self.data = json.load(f)
            self.transform = transform_local

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            image = Image.open(item["image"])
            label = item["label"]
            return self.transform(image), label

    if os.path.exists("feedback_labels.json"):
        feedback_data = FeedbackDataset("feedback_labels.json")
        from torch.utils.data import ConcatDataset
        train_data = ConcatDataset([mnist_data, feedback_data])
    else:
        train_data = mnist_data

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    retrain_model = CNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(retrain_model.parameters(), lr=0.001)

    for epoch in range(3):  # fast retrain, you can increase if needed
        retrain_model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = retrain_model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"📦 Epoch {epoch+1}: Loss {total_loss / len(train_loader):.4f}")

    torch.save(retrain_model.state_dict(), MODEL_PATH)
    print("✅ Model retrained and saved.")

if __name__ == "__main__":
    app.run(debug=True)
