from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from model import chain  # Import the chain from model.py

app = FastAPI()

@app.get("/query")
async def query(q: str = Query(..., description="Your question")):
    async def stream():
        answer = chain.invoke(q)
        yield answer
    return StreamingResponse(stream(), media_type="text/plain")