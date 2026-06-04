from __future__ import annotations

import io
import logging

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.core.config import CLASS_LABELS
from app.core.model import ModelManager
from app.schemas.prediction import PredictionResponse

logger = logging.getLogger(__name__)

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class ImageProcessingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class PredictorService:
    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        if not image_bytes:
            raise ImageProcessingError("Nenhuma imagem foi enviada.")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            image = image.convert("RGB")
            tensor = INFERENCE_TRANSFORM(image)
            return tensor.unsqueeze(0)
        except UnidentifiedImageError as exc:
            raise ImageProcessingError(
                "Arquivo corrompido ou formato de imagem inválido."
            ) from exc
        except ImageProcessingError:
            raise
        except Exception as exc:
            logger.exception("Erro ao processar imagem")
            raise ImageProcessingError(
                "Não foi possível processar a imagem enviada."
            ) from exc

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        model = self._model_manager.model
        device = self._model_manager.device

        input_tensor = self.preprocess(image_bytes).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence_value, predicted_index = torch.max(probabilities, dim=1)

        index = int(predicted_index.item())
        confidence_percent = round(float(confidence_value.item()) * 100, 2)
        prediction = CLASS_LABELS[index]

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence_percent,
        )
