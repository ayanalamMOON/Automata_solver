import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test fixtures and configurations here
import pytest
from fastapi.testclient import TestClient

# Create a fixture that will be used by integration tests
@pytest.fixture
def integration_client():
    """Test client fixture for integration tests"""
    # Only import the app when running integration tests
    from main import app
    return TestClient(app)