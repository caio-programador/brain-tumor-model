import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE_BYTES
from app.dependencies import get_predictor_service
from app.schemas.prediction import ErrorResponse, PredictionResponse, StatusResponse
from app.services.predictor import ImageProcessingError, PredictorService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/",
    response_model=StatusResponse,
    summary="Verificar status da API",
    tags=["Health"],
)
async def health_check() -> StatusResponse:
    return StatusResponse(status="online")


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
    summary="Classificar tumor cerebral",
    tags=["Prediction"],
)
async def predict_tumor(
    image: UploadFile = File(..., description="Imagem de ressonância magnética T1C+"),
    service: PredictorService = Depends(get_predictor_service),
) -> PredictionResponse:
    if image.filename is None or image.filename.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nenhuma imagem foi enviada.",
        )

    content_type = image.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato inválido. Envie apenas imagens PNG ou JPEG.",
        )

    try:
        image_bytes = await image.read()
    except Exception as exc:
        logger.exception("Erro ao ler arquivo enviado")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Não foi possível ler o arquivo enviado.",
        ) from exc

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nenhuma imagem foi enviada.",
        )

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A imagem excede o limite de 10 MB.",
        )

    try:
        return service.predict(image_bytes)
    except ImageProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        ) from exc
    except Exception as exc:
        logger.exception("Erro interno durante inferência")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar a imagem. Tente novamente mais tarde.",
        ) from exc
