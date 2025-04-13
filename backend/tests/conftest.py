import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test fixtures and configurations here
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)