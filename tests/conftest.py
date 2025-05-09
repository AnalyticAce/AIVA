"""
Test fixtures and configuration for API tests.
"""
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """
    Create a test client for the FastAPI application.
    
    Returns:
        TestClient: A test client for the API
    """
    return TestClient(app)