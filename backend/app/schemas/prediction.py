from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    status: str = Field(examples=["online"])


class PredictionResponse(BaseModel):
    prediction: str = Field(examples=["Glioma"])
    confidence: float = Field(examples=[97.85], description="Confiança em percentual")


class ErrorResponse(BaseModel):
    message: str = Field(examples=["Formato de imagem inválido."])
