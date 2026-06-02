import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes, device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    return model
