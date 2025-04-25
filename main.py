from utils.logging import setup_logger
from utils.analyze_query import analyze_query
from schemas import ResearchRequest, ResearchResponse
from tools.web_search_tool import get_google_search_tool
from tools.web_scraper import allowed_to_scrape, fetch_page, parse_html
from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn
from utils.get_embeddings import get_embeddings
import os
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.get_llm import get_llm



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

    # query analysis
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
        logger.info(f"urls found: ({len(urls)}")
        snippets = [r.get("snippet") for r in results]
        logger.info(f"{sq} => URLs: {urls}")

         # TODO: ranking and relevance of urls (maybe just top 5)
        for url in urls:
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
    logger.info(f"Scraped: {len(docs)}")

    # info synthesis
    embeddings = get_embeddings()
    logger.info(f"embeddings created #{embeddings}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
    )

    documents = []
    for raw in docs:
        # create a Document with metadata
        doc = Document(page_content=raw["content"], metadata={"source": raw["url"]})
        # split into chunks
        for chunk in text_splitter.split_documents([doc]):
            documents.append(chunk)

    logger.info(f"Documents created #{documents}")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    print("Vectorstore created", vectorstore)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_chunks = retriever.get_relevant_documents(query_analysis.get("search_strategy", query))

    # solve contradictions
    print('relevant chunks', relevant_chunks)
    llm = get_llm()
    system_content = (
        "You are an AI research assistant. Your task is to synthesize information and produce a coherent summary. "
        "Follow these guidelines:\n"
        "1. Combine information from multiple context chunks.\n"
        "2. Identify and resolve any contradictory information.\n"
        "3. Organize the information into a clear, logical structure.\n"
        "4. Generate a concise summary that directly answers the user's original query.\n"
        "5. Cite sources inline using the provided URLs." 
        f"6. Adhere to the user's intent: {query_analysis.get('intent', 'research')} and deliver the type of information requested: {query_analysis.get('information_type', 'research')}.\n"
    )
    context_text = "\n\n".join(
        f"Source: {chunk.metadata['source']}\n{chunk.page_content}" 
        for chunk in relevant_chunks
    )

    user_prompt = (
        "Below are excerpts from web sources. Use them in conjunction with the analysis context to answer the question.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\n"
        "Please include relevant facts, figures, and data in tabular format where applicable."
    )

    response = llm.chat.completions.create(messages=[{"role": "system", "content": system_content},{"role": "user", "content": user_prompt}], model='gpt-4o-mini-2')
    print('Final Response from LLM', response)
    answer = response.choices[0].message.content
    sources = list({c.metadata['source'] for c in relevant_chunks})

    return {
        "query": query,
        "result": {"content": answer, "sources": sources }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
