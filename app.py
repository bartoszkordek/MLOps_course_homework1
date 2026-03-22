from fastapi import FastAPI

from api.models.schemas import PredictRequest, PredictResponse

app = FastAPI()


@app.get("/")
def welcome_root() -> dict[str, str]:
    return {"message": "Welcome to the ML API"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    prediction_result = "positive"
    return PredictResponse(prediction=prediction_result)
