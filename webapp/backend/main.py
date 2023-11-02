import uvicorn as uvicorn
from fastapi import FastAPI
import os

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        reload=os.getenv("RELOAD", True),
        port=int(os.getenv("PORT", 8080))
    )
