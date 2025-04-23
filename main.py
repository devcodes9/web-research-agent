from utils.logging import setup_logger
from schemas import ResearchRequest, ResearchResponse
from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
load_dotenv()

logger = setup_logger("web-research-agent")

app = FastAPI(title="Web Research Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error occurred during request: {request.url} - {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/execute-research", response_model=ResearchResponse)
async def execute_research(payload: ResearchRequest = Body(...)):
    query = payload.query
    result = {}
    return {"query": query, "result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
