import uvicorn as uvicorn
from fastapi import FastAPI
import os
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



##### Pydantic models #####
class PredictionRequest(BaseModel):
    type: list
    index: list


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#### Machine Learning
decision_tree = joblib.load('webapp/backend/models/decision_tree.joblib')
nearest_centroid = joblib.load('webapp/backend/models/nearest_centroid.joblib')
options = {0: "no", 1: "yes"}
party = {0: 'AfD', 1: 'CDU', 2: 'CSU', 3: 'FDP', 4: 'GRÜNE', 5: 'Linke', 6: 'SPD'}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(body: PredictionRequest):
    questions = np.full(244, "no", dtype=object).reshape(1, -1)
    user_response = body.type
    question_index = body.index
    answers = [options[v] for v in user_response]
    for a, i in zip(answers, question_index):
        questions[0][i] = a
    prediction = decision_tree.predict(questions)[0]
    return {"prediction": party[prediction]}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        reload=os.getenv("RELOAD", True),
        port=int(os.getenv("PORT", 8080))
    )
