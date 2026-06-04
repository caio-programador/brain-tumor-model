from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

CLASS_LABELS: tuple[str, ...] = (
    "Astrocytoma",
    "Ependymoma",
    "Glioma",
    "Hemangiopericytoma",
    "Meningioma",
    "Neurocytoma",
    "Normal",
    "Oligodendroglioma",
    "Other",
    "Schwannoma",
)

ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/jpg"}
)

MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024


class Settings(BaseModel):
    model_path: Path = Field(default=BASE_DIR / "models" / "best_model.pth")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    @classmethod
    def from_env(cls) -> "Settings":
        import os

        model_path_raw = os.getenv("MODEL_PATH", "models/best_model.pth")
        model_path = Path(model_path_raw)
        if not model_path.is_absolute():
            model_path = BASE_DIR / model_path

        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))

        return cls(model_path=model_path, host=host, port=port)


def get_settings() -> Settings:
    return Settings.from_env()
