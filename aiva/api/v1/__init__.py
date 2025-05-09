"""
API router initialization for version 1 of the API.
"""
from fastapi import APIRouter

from aiva.api.v1.endpoints import agent

# Create the API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])