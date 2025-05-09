"""
AIVA (AI Virtual Assistant) API Server
"""
import contextlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aiva.api.v1.endpoints import agent
from aiva.core.config import settings
from aiva.core.logging import logger, configure_logging

# Configure logging
configure_logging()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    Handles startup and shutdown events for the application.
    
    Args:
        app: FastAPI application instance
    """
    # Startup actions
    logger.info(f"Starting AIVA API server v{app.version}")
    logger.info(f"Environment: {settings.environment}")
    
    yield
    
    # Shutdown actions
    logger.info("Shutting down AIVA API server")


# Create FastAPI application
app = FastAPI(
    title="AIVA Finance API",
    description="AI-powered finance assistant API for processing financial data",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(agent.router, prefix="/api/v1")


@app.get("/")
async def root():
    """
    Root endpoint providing basic API information in JSON format.
    
    Returns:
        JSONResponse: API information including name, version, description and endpoints
    """
    return JSONResponse({
        "name": "AIVA Finance API",
        "version": app.version,
        "description": "AI-powered finance assistant API for processing financial data",
        "documentation": "/docs",
        "endpoints": {
            "process_financial_prompt": "/api/v1/agent/process",
        },
        "status": "operational",
        "environment": settings.environment,
        "maintainer": "AnalyticAce"
    })


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting AIVA with uvicorn on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
