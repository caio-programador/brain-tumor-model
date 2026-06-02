import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from pathlib import Path
import shutil

SOURCE = Path("dataset")
TARGET = Path("dataset_t1c")

TARGET.mkdir(exist_ok=True)

def rename_files():
  for folder in SOURCE.iterdir():

    if "T1C+" not in folder.name:
      continue

    tumor_name = folder.name.replace(" T1C+", "")

    destination = TARGET / tumor_name

    shutil.copytree(
      folder,
      destination,
      dirs_exist_ok=True
    )

rename_files()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Transformações para aumentar o dataset e normalizar as imagens
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])
# Dataset/Pasta por classe
train_ds = ImageFolder('dataset_t1c', transform=transform)
# DataLoader para batch e shuffle
train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    pin_memory=True,
    num_workers=4
)
# Modelo pré-treinado
weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)
num_classes = len(train_ds.classes)

model.fc = nn.Linear(
  model.fc.in_features,
  num_classes
)
model.to(device)
# Otimizador e loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
best_acc = 0
for epoch in range(10):
  model.train()
  
  correct = 0
  total = 0
  running_loss = 0

  for imgs, labels in train_loader:
    imgs, labels = imgs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(imgs)
    
    _, preds = torch.max(
      outputs,
      1
    )
    correct += (
      preds == labels
    ).sum().item()
    
    total += labels.size(0)

    loss = criterion(
        outputs,
        labels
    )
    

    loss.backward()
    running_loss += loss.item()
    optimizer.step()
    
    
  accuracy = 100 * correct / total

  print(
    f"Epoch {epoch+1} "
    f"| Loss {running_loss:.4f} "
    f"| Accuracy {accuracy:.2f}%"
  )
  
  if accuracy > best_acc:

    best_acc = accuracy

    torch.save(
        model.state_dict(),
        "best_model.pth"
    )
