import csv
import json
import os
import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Adiciona a raiz do projeto para importarmos o 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import TEST_DIR, TRAIN_DIR, get_transforms, prepare_dataset
from core.model import get_model

MODEL_PATH = "best_model.pth"
OUTPUT_DIR = Path("article_artifacts")
RAW_MATRIX_IMAGE = OUTPUT_DIR / "confusion_matrix_raw.png"
NORMALIZED_MATRIX_IMAGE = OUTPUT_DIR / "confusion_matrix_normalized.png"
REPORT_FILE = OUTPUT_DIR / "classification_report.txt"
PREDICTIONS_FILE = OUTPUT_DIR / "test_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "metrics.json"


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Garante que o dataset físico exista para podermos carregar as classes
    prepare_dataset()

    OUTPUT_DIR.mkdir(exist_ok=True)

    train_dataset = ImageFolder(str(TRAIN_DIR))
    test_dataset = ImageFolder(str(TEST_DIR), transform=get_transforms()[1])

    classes = train_dataset.classes
    print("Classes encontradas:")
    print(classes)

    if not os.path.exists(MODEL_PATH):
        print(f"Erro: Modelo não encontrado no caminho {MODEL_PATH}")
        return

    model = get_model(len(classes), device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels_array = np.array(all_labels)
    all_preds_array = np.array(all_preds)
    all_probs_array = np.array(all_probs)

    cm = confusion_matrix(all_labels_array, all_preds_array)
    cm_normalized = confusion_matrix(all_labels_array, all_preds_array, normalize="true")
    accuracy = accuracy_score(all_labels_array, all_preds_array)
    macro_f1 = f1_score(all_labels_array, all_preds_array, average="macro")
    weighted_f1 = f1_score(all_labels_array, all_preds_array, average="weighted")
    report_text = cast(str, classification_report(all_labels, all_preds, target_names=classes, digits=4))

    print("\nRelatório de classificação")
    print("--------------------------")
    print(report_text)
    print(f"Acurácia no teste: {accuracy * 100:.2f}%")
    print(f"F1 macro: {macro_f1:.4f}")
    print(f"F1 ponderado: {weighted_f1:.4f}")

    with open(REPORT_FILE, "w", encoding="utf-8") as report_handle:
        report_handle.write(report_text)
        report_handle.write(f"\nAcurácia no teste: {accuracy * 100:.2f}%\n")
        report_handle.write(f"F1 macro: {macro_f1:.4f}\n")
        report_handle.write(f"F1 ponderado: {weighted_f1:.4f}\n")

    metrics_payload = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "num_test_samples": int(len(all_labels_array)),
        "num_classes": int(len(classes)),
        "classes": classes,
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as metrics_handle:
        json.dump(metrics_payload, metrics_handle, indent=2, ensure_ascii=False)

    with open(PREDICTIONS_FILE, "w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.writer(csv_handle)
        header = ["filename", "true_label", "predicted_label", "confidence"]
        header.extend([f"prob_{class_name}" for class_name in classes])
        writer.writerow(header)

        for sample, true_label, pred_label, probs_row in zip(
            test_dataset.samples,
            all_labels_array,
            all_preds_array,
            all_probs_array,
        ):
            file_path, _ = sample
            confidence = float(probs_row[int(pred_label)])
            writer.writerow(
                [
                    Path(file_path).name,
                    classes[int(true_label)],
                    classes[int(pred_label)],
                    f"{confidence:.6f}",
                    *[f"{float(prob):.6f}" for prob in probs_row],
                ]
            )

    disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)

    fig_raw, ax_raw = plt.subplots(figsize=(14, 12))
    disp_raw.plot(ax=ax_raw, cmap="Blues", xticks_rotation=45, values_format="d", colorbar=True)
    ax_raw.set_title("Matriz de Confusão - Valores Absolutos")
    plt.tight_layout()
    plt.savefig(RAW_MATRIX_IMAGE, dpi=220, bbox_inches="tight")
    plt.close(fig_raw)

    fig_norm, ax_norm = plt.subplots(figsize=(14, 12))
    disp_norm.plot(ax=ax_norm, cmap="Blues", xticks_rotation=45, values_format=".2f", colorbar=True)
    ax_norm.set_title("Matriz de Confusão - Normalizada por Classe")
    plt.tight_layout()
    plt.savefig(NORMALIZED_MATRIX_IMAGE, dpi=220, bbox_inches="tight")
    plt.close(fig_norm)

    print(f"\nMatriz de confusão salva em: {RAW_MATRIX_IMAGE}")
    print(f"Matriz normalizada salva em: {NORMALIZED_MATRIX_IMAGE}")
    print(f"Relatório salvo em: {REPORT_FILE}")
    print(f"Métricas salvas em: {METRICS_FILE}")
    print(f"Predições por imagem salvas em: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    evaluate_model()