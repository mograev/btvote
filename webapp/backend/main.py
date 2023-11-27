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
import matplotlib.pyplot as plt
import io




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
file = open('./models/classifier.pkl', 'rb')
classifier = pickle.load(file)
options = {0: "no", 1: "yes"}
party = {0: 'AFD', 1: 'CDU/CSU', 2: 'FDP', 3: 'GRÜNE', 4: 'Linke', 5: 'SPD'}
party_colors = {0: '#1592bf', 1: 'black', 2: 'yellow', 3: '#067808', 4: '#d408bf', 5: 'red'}
results = {i: 0 for i in range(6)}

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(body: PredictionRequest):
    user_response = body.type
    user_response = np.array(user_response).reshape(1, -1)
    prediction = classifier.predict_proba(user_response)[0].tolist()
    pred_class = classifier.predict(user_response)[0]
    results[pred_class] += 1
    sorted_result = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    values = [value for key, value in sorted_result.items()]
    parties = [party[key] for key in sorted_result]
    colors = [party_colors[key] for key in sorted_result]

    total = sum(values)
    percentages = [100 * value / total for value in values]

    plt.bar(range(6), values, color=colors)
    plt.xticks(range(6), parties)

    for i, (value, percentage) in enumerate(zip(values, percentages)):
        plt.text(i, value + 0.01, f'{percentage:.2f}%', ha='center')

    plt.savefig('./foo.png')
    plt.close()
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        reload=os.getenv("RELOAD", True),
        port=int(os.getenv("PORT", 8081))
    )
