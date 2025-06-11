import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
from matplotlib import pyplot as plt

# Configuration
BATCH_SIZE =64
EPOCHS = 5
MODEL_PATH = "mnistnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),              # Rescale to 28x28
    transforms.Grayscale(num_output_channels=1),  # Convert to 1-channel grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])



class MNISTNN(nn.Module):
    def __init__(self):
        super(MNISTNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x =(self.fc2(x))
        return x

class Pipeline:
    def __init__(self, transform, model_path="mnistnn.pth", mode="train", epochs=EPOCHS):
        self.transform = transform
        self.model_path = model_path
        self.mode = mode
        self.epochs = epochs
        if self.mode == "train":
    
            # Datasets
            train_val_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)

            # Split train/validation
            train_size = int(0.8 * len(train_val_dataset))
            val_size = len(train_val_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(train_val_dataset, [train_size, val_size])

            # Dataloaders
            self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            #self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


        self.model = MNISTNN().to(DEVICE)
        self.model.train()
        if self.mode == "inference":
            self.load()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0006)
        self.criterion = nn.CrossEntropyLoss()
    
    def load(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        print("\033[91mModel loaded.\033[0m")
    def save(self):
        torch.save(self.model.state_dict(), self.model_path)
        print("\033[94mModel saved.\033[0m")

    def train(self):
        # Training
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            num_train = 0
            for _, (data, target) in enumerate(self.train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_train += target.size(0)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/num_train:.4f}")

            # Validation
            self.model.eval()
            correct, num_val = 0, 0
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = self.model(data)
                    pred = output.argmax(dim=1, keepdim=False)
                    for i in range(pred.size(0)):
                        correct += (pred[i] == target[i]).item()
                        num_val += 1
            print(f"Validation Accuracy: {100. * correct / num_val:.2f}%")
        
    def inference(self, img, bkg='white'):
        self.model.eval()
        with torch.no_grad():
            #img should be flipped if background is white
            if bkg == 'white':
                data= 1 - self.transform(img)
            else:
                data = self.transform(img)
            data = data.to(DEVICE)
            output = self.model(data)
            pred = output.argmax(dim=1, keepdim=False)
            return  pred.item()

if __name__ == "__main__":
    pipeline = Pipeline(transform)
    import os
    if not os.path.exists(MODEL_PATH):
        pipeline.train()
        pipeline.save()
    pipeline.load()
    import requests
    from io import BytesIO

    url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQNAJo2MDPQF53ZfegRG9xBSoKF_a-9GWgKxQ&s'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save('mnist_test.png')
    result = pipeline.inference(img)
    print("The predicted digit is:", result)
