import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Adiciona a raiz do projeto para conseguirmos importar o módulo 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import get_dataloaders
from core.model import get_model

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Preparando dados...")
    train_loader, val_loader, classes = get_dataloaders(batch_size=16)
    
    print(f"Classes identificadas: {len(classes)}")
    
    model = get_model(len(classes), device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0

    print("Iniciando treinamento...")
    for epoch in range(10):
        # === FASE DE TREINAMENTO ===
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

        # === FASE DE VALIDAÇÃO ===
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

if __name__ == "__main__":
    train()
