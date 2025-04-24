from utils.logging import setup_logger
from utils.analyze_query import analyze_query
from schemas import ResearchRequest, ResearchResponse
from tools.web_search_tool import get_google_search_tool
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

    # TODO: query analysis
    query_analysis = analyze_query(query)
    logger.info(f"Query analysis: {query_analysis}")
    # web search
    # google_search_tool = get_google_search_tool()
    # search_results = google_search_tool.func(query, 10)
    # logger.info(f"Google search results: {search_results}")

    # TODO: ranking and relevance


    # TODO: content extraction

    # TODO: info synthesis

    result = {
        "content": 'test',
        "score": 0.95  # Replace with actual score
    }

    return {"query": query, "result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
