import torch
import sys
import os
from PIL import Image

# Adiciona a raiz do projeto para importarmos o 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import get_transforms, prepare_dataset, TRAIN_DIR
from core.model import get_model
from torchvision.datasets import ImageFolder

MODEL_PATH = "best_model.pth"

def analyze_image():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Garante que o dataset físico exista para podermos carregar as classes
    prepare_dataset()
    dataset = ImageFolder(str(TRAIN_DIR))
    classes = dataset.classes

    print("Classes encontradas:")
    print(classes)

    # Pegar somente o transform sem randomização (val_transform)
    _, val_transform = get_transforms()

    model = get_model(len(classes), device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"Erro: Modelo não encontrado no caminho {MODEL_PATH}")
        return

    model.eval()

    while True:
        image_path = input("\nDigite o caminho da imagem (ou 'sair' para encerrar): ")

        if image_path.lower() == "sair":
            break
        
        if not os.path.exists(image_path):
            print("Arquivo não encontrado!")
            continue

        try:
            image = Image.open(image_path)
            image = val_transform(image)
            image = image.unsqueeze(0) # type: ignore
            image = image.to(device)

            with torch.no_grad():
                outputs = model(image)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred = torch.max(probs, 1)

            predicted_class = classes[pred.item()] # type: ignore

            print("\nResultado")
            print("--------------------")
            print("Classe:", predicted_class)
            print("Confiança:", f"{confidence.item()*100:.2f}%")
        except Exception as e:
            print("Erro ao processar a imagem:", e)

if __name__ == "__main__":
    analyze_image()
