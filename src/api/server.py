"""
PeopleQuery AI - FastAPI Application Server
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

app = FastAPI(
    title="PeopleQuery AI",
    description="Enterprise HR Intelligence & Workforce Analytics Copilot API",
    version="1.0.0",
)

# Enable CORS for frontend applications (Vite default port 5173, Next default 3000, preview 4173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:4173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "app": "PeopleQuery AI",
        "status": "online",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "docs_list": "/api/docs/list",
            "chat": "/api/chat",
            "chat_stream": "/api/chat/stream",
        },
    }
