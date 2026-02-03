from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from engine import QueryBatchEngine
import asyncio
from config import settings

app = FastAPI()
engine = QueryBatchEngine(model_path=settings.MODEL_PATH)

class QueryRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(engine.batch_worker())

@app.post("/v1/query-classify")
async def query_classify(request: QueryRequest, response: Response):
    res = await engine.classify(request.text)
    label = res['label']
    confidence = res['confidence']
    if label == 1 and confidence < settings.THRESHOLD:
        label = 0

    if label == 1:
        await asyncio.sleep(settings.DELAY_SLOW) 
    else:
        await asyncio.sleep(settings.DELAY_FAST)
    
    response.headers["x-router-latency"] = "10"
    
    return {"label": label}