import torch
from torchvision import transforms
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

def prepare_dataset():
    TARGET.mkdir(exist_ok=True)
    if TRAIN_DIR.exists():
        return

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

def get_transforms():
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

    # Transformações sem randomização (para validação e predição)
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5]
        )
    ])
    
    return train_transform, val_transform

def get_dataloaders(batch_size=16):
    prepare_dataset()
    
    train_transform, val_transform = get_transforms()
    
    train_ds = ImageFolder(str(TRAIN_DIR), transform=train_transform)
    val_ds = ImageFolder(str(VAL_DIR), transform=val_transform)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    
    return train_loader, val_loader, train_ds.classes
