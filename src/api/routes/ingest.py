"""
/ingest endpoint: trigger scrape -> clean -> chunk -> embed -> index for a given URL.
"""
from fastapi import APIRouter
from src.api.schemas import IngestRequest, IngestResponse

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/", response_model=IngestResponse)
async def ingest_url(payload: IngestRequest):
    # TODO: call scraper -> preprocessing -> embeddings pipeline
    raise NotImplementedError("Wire up the ingestion pipeline here.")
