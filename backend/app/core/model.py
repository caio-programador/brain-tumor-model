import logging

import torch
import torch.nn as nn
from torchvision import models

from app.core.config import CLASS_LABELS, Settings

logger = logging.getLogger(__name__)


def build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Dispositivo selecionado: CUDA (%s)", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Dispositivo selecionado: CPU")
    return device


class ModelManager:
    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._device: torch.device | None = None
        self._loaded: bool = False

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Modelo não carregado. Inicialize a aplicação primeiro.")
        return self._model

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("Dispositivo não configurado. Inicialize a aplicação primeiro.")
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, settings: Settings) -> None:
        if self._loaded:
            return

        if not settings.model_path.exists():
            raise FileNotFoundError(
                f"Arquivo do modelo não encontrado: {settings.model_path}"
            )

        device = resolve_device()
        model = build_resnet18(len(CLASS_LABELS))
        state_dict = torch.load(settings.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self._model = model
        self._device = device
        self._loaded = True

        logger.info("Modelo carregado com sucesso: %s", settings.model_path)


model_manager = ModelManager()
