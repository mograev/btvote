import uvicorn as uvicorn
from fastapi import FastAPI
import os
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle



##### Pydantic models #####
class PredictionRequest(BaseModel):
    type: list


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#### Machine Learning
#decision_tree = joblib.load('webapp/backend/models/decision_tree.joblib')
#random_forest = joblib.load('webapp/backend/models/random_forest.joblib')
file = open('./webapp/backend/models/classifier.pkl', 'rb')
classifier = pickle.load(file)
options = {0: "no", 1: "yes"}
party = {0: 'AfD', 1: 'CDU', 2: 'FDP', 3: 'GRÜNE', 4: 'Linke', 5: 'SPD'}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(body: PredictionRequest):
    user_response = body.type
    user_response = np.array(user_response).reshape(1, -1)
    prediction = classifier.predict_proba(user_response)[0].tolist()
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        reload=os.getenv("RELOAD", True),
        port=int(os.getenv("PORT", 8081))
    )
