import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

# -----------------------
# CONFIG
# -----------------------

IMAGE_PATH = "image.png"
MODEL_PATH = "best_model.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------
# CLASSES
# -----------------------

dataset = ImageFolder("dataset_t1c")

classes = dataset.classes

print("Classes encontradas:")
print(classes)

# -----------------------
# TRANSFORM
# -----------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# -----------------------
# MODELO
# -----------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

# -----------------------
# IMAGEM
# -----------------------

image = Image.open(
    IMAGE_PATH
)

image = transform(image)

image = image.unsqueeze(0) # type: ignore

image = image.to(device)

# -----------------------
# PREDIÇÃO
# -----------------------

with torch.no_grad():

    outputs = model(image)

    probs = torch.softmax(
        outputs,
        dim=1
    )

    confidence, pred = torch.max(
        probs,
        1
    )

predicted_class = classes[
    pred.item()
] # type: ignore

print("\nResultado")
print("--------------------")
print("Classe:", predicted_class)
print(
    "Confiança:",
    f"{confidence.item()*100:.2f}%"
)