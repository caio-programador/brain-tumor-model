import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from pathlib import Path
import shutil
import random

SOURCE = Path("dataset")
TARGET = Path("dataset_t1c")

TRAIN_DIR = TARGET / "train"
VAL_DIR = TARGET / "val"
TEST_DIR = TARGET / "test"

TARGET.mkdir(exist_ok=True)

def rename_files():
  random.seed(42) # garante que as imagens separadas sempre sejam as mesmas
  for folder in SOURCE.iterdir():

    if "T1C+" not in folder.name:
      continue

    tumor_name = folder.name.replace(" T1C+", "")
    
    # Pegar todos os arquivos da pasta e embaralhar
    files = [f for f in folder.iterdir() if f.is_file()]
    random.shuffle(files)
    
    # Calcular 70% para o treino, 15% pra validação e 15% para teste manual
    train_size = int(0.7 * len(files))
    val_size = int(0.15 * len(files))
    
    train_files = files[:train_size]
    val_files = files[train_size:train_size + val_size]
    test_files = files[train_size + val_size:]

    # Criar pastas de treino, validação e teste pros arquivos
    train_dest = TRAIN_DIR / tumor_name
    val_dest = VAL_DIR / tumor_name
    test_dest = TEST_DIR / tumor_name
    
    train_dest.mkdir(parents=True, exist_ok=True)
    val_dest.mkdir(parents=True, exist_ok=True)
    test_dest.mkdir(parents=True, exist_ok=True)

    # Copiar fisicamente para as pastas separadas
    for f in train_files:
      shutil.copy2(f, train_dest / f.name)
    for f in val_files:
      shutil.copy2(f, val_dest / f.name)
    for f in test_files:
      shutil.copy2(f, test_dest / f.name)

if not TRAIN_DIR.exists():
  rename_files()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformações com data augmentation extra (para treino)
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

# Transformações sem randomização (para validação)
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

# Datasets já lendo diretamente de cada pasta física que criamos!
train_ds = ImageFolder(str(TRAIN_DIR), transform=train_transform)
val_ds = ImageFolder(str(VAL_DIR), transform=val_transform)

# DataLoader para batch e shuffle
train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    pin_memory=True,
    num_workers=4
)

val_loader = DataLoader(
    val_ds,
    batch_size=16,
    shuffle=False,
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
best_val_acc = 0

for epoch in range(10):
  #=== FASE DE TREINAMENTO ===
  model.train()
  
  train_correct = 0
  train_total = 0
  train_loss = 0

  for imgs, labels in train_loader:
    imgs, labels = imgs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(imgs)
    
    _, preds = torch.max(outputs, 1)
    train_correct += (preds == labels).sum().item()
    train_total += labels.size(0)

    loss = criterion(outputs, labels)
    
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
    
  train_accuracy = 100 * train_correct / train_total
  avg_train_loss = train_loss / len(train_loader)

  #=== FASE DE VALIDAÇÃO ===
  model.eval()
  val_correct = 0
  val_total = 0
  val_loss = 0

  with torch.no_grad():
    for imgs, labels in val_loader:
      imgs, labels = imgs.to(device), labels.to(device)
      outputs = model(imgs)

      _, preds = torch.max(outputs, 1)
      val_correct += (preds == labels).sum().item()
      val_total += labels.size(0)
      
      loss = criterion(outputs, labels)
      val_loss += loss.item()
      
  val_accuracy = 100 * val_correct / val_total
  avg_val_loss = val_loss / len(val_loader)

  print(
    f"Epoch {epoch+1:02d}/10 "
    f"| Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% "
    f"| Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
  )
  
  # Salvar melhor modelo com base na validação
  if val_accuracy > best_val_acc:
    best_val_acc = val_accuracy
    torch.save(
        model.state_dict(),
        "best_model.pth"
    )
