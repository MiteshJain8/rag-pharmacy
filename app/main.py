from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.v1.routes import router as v1_router
from app.core.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.app_version)
app.include_router(v1_router)
UI_PATH = Path(__file__).parent / "web" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> HTMLResponse:
    return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}
