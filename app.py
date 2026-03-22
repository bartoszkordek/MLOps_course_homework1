from fastapi import FastAPI

import inference
from api.models.schemas import PredictRequest, PredictResponse

app = FastAPI()

classifier_model = inference.load_classifier_model()
sentence_transformer_model = inference.load_sentence_transformer_model()


@app.get("/")
def welcome_root() -> dict[str, str]:
    return {"message": "Welcome to the ML API"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    embedding = inference.transform(sentence_transformer_model, request.text)
    prediction_result = inference.predict(classifier_model, embedding)
    return PredictResponse(prediction=prediction_result)
