import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test fixtures and configurations here
import pytest
from fastapi.testclient import TestClient
from typing import Generator
import redis
from main import app
from security import create_access_token

# Create a fixture that will be used by integration tests
@pytest.fixture
def client() -> Generator:
    with TestClient(app) as c:
        yield c

@pytest.fixture
def test_token() -> str:
    """Create a test token for protected endpoints"""
    access_token = create_access_token(data={"sub": "johndoe"})
    return access_token

@pytest.fixture
def authorized_client(client: TestClient, test_token: str) -> TestClient:
    """Create an authorized test client"""
    client.headers = {
        **client.headers,
        "Authorization": f"Bearer {test_token}"
    }
    return client