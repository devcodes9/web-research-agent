from utils.logging import setup_logger
from utils.analyze_query import analyze_query
from schemas import ResearchRequest, ResearchResponse
from tools.web_search_tool import get_google_search_tool
from tools.web_scraper import allowed_to_scrape, fetch_page, parse_html
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
    subqueries = query_analysis.get("subqueries", [query])

    # web search
    google_search_tool = get_google_search_tool()
    # search_results = google_search_tool.func(query, 10)
    # logger.info(f"Google search results: {search_results}")

    
    docs = []
    for sq in subqueries:
        results = google_search_tool.func(sq, num_results=10)
        urls = [r["link"] for r in results]
        snippets = [r.get("snippet") for r in results]
        logger.info(f"{sq} => URLs: {urls}")

         # TODO: ranking and relevance of urls
        for url in urls[:3]:
            if not await allowed_to_scrape(url):
                logger.warning(f"Blocked by robots.txt: {url}")
                continue
            try:
                html = await fetch_page(url, 1)
                text = parse_html(html)
                # if not moderate_content(text):
                #     logger.warning(f"Content flagged: {url}")
                #     continue
                docs.append({"url": url, "content": text, "score": len(text)})
            except Exception as e:
                logger.warning(e)
    logger.info(f"Scraped: {docs}")

    # TODO: info synthesis
    
    result = {
        "content": 'test',
        "score": 0.95  # Replace with actual score
    }

    return {"query": query, "result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
