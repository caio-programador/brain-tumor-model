from app.core.model import model_manager
from app.services.predictor import PredictorService

predictor_service = PredictorService(model_manager)


def get_predictor_service() -> PredictorService:
    return predictor_service
