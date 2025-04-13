import pytest
from fastapi.testclient import TestClient
from main import app
import json
import redis
import os
from dotenv import load_dotenv

# Load environment variables for testing
load_dotenv()

# Initialize test client
client = TestClient(app)

# Initialize Redis test client
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=1  # Use different DB for testing
)

@pytest.fixture(autouse=True)
def clear_redis():
    """Clear Redis test database before each test"""
    redis_client.flushdb()
    yield
    redis_client.flushdb()

@pytest.fixture
def authorized_client():
    """Fixture for authorized client"""
    # Get token using test credentials
    response = client.post("/token", data={
        "username": "johndoe",
        "password": "secret"
    })
    assert response.status_code == 200
    token = response.json()["access_token"]
    return TestClient(app, headers={"Authorization": f"Bearer {token}"})

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_detailed_health_check(client):
    """Test the detailed health check endpoint"""
    response = client.get("/health/detailed")
    assert response.status_code == 200
    data = response.json()
    assert "components" in data
    assert "redis" in data["components"]
    assert "system" in data["components"]

def test_metrics_endpoint(client):
    """Test the Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"http_requests_total" in response.content

def test_dfa_conversion(client):
    """Test DFA conversion endpoint"""
    test_data = {
        "regex": "a*b"
    }
    response = client.post("/convert", json=test_data)
    assert response.status_code == 200
    assert "dfa_svg" in response.json()

def test_batch_processing(authorized_client):
    """Test batch processing endpoint"""
    test_data = {
        "items": [
            {
                "regex": "a*b",
                "test_cases": [
                    {"input": "ab", "expected": True},
                    {"input": "c", "expected": False}
                ]
            }
        ],
        "parallel": True
    }
    response = authorized_client.post("/api/bulk/convert", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["success_count"] > 0

def test_automata_analysis(authorized_client):
    """Test automata analysis endpoint"""
    test_automaton = {
        "states": ["q0", "q1"],
        "alphabet": ["a", "b"],
        "transitions": {
            "q0": {"a": "q0", "b": "q1"},
            "q1": {"a": "q1", "b": "q1"}
        },
        "start_state": "q0",
        "accept_states": ["q1"]
    }
    
    test_data = {
        "automaton": test_automaton,
        "properties": ["deterministic", "minimal"]
    }
    
    response = authorized_client.post("/api/analyze", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "deterministic" in data

def test_error_handling(client):
    """Test error handling for invalid input"""
    test_data = {
        "regex": "("  # Invalid regex
    }
    response = client.post("/convert", json=test_data)
    assert response.status_code == 400
    assert "error" in response.json() or "detail" in response.json()

def test_rate_limiting(client):
    """Test rate limiting middleware"""
    # Make a small batch of requests quickly
    test_batch_size = 5
    responses = []
    
    # Make initial requests
    for _ in range(test_batch_size):
        responses.append(client.get("/health"))
        
    # All initial requests should succeed
    assert all(r.status_code == 200 for r in responses)
    
    # Now make many requests quickly to trigger rate limit
    for _ in range(60):  # Force rate limit
        responses.append(client.get("/health"))
    
    # At least one request should be rate limited
    assert any(r.status_code == 429 for r in responses)

def test_caching(client):
    """Test Redis caching for automata explanations"""
    test_query = "What is a DFA?"
    
    # First request should hit the API
    response1 = client.get(f"/explain/{test_query}")
    assert response1.status_code == 200
    
    # Second request should hit the cache
    response2 = client.get(f"/explain/{test_query}")
    assert response2.status_code == 200
    
    # Both responses should be identical
    assert response1.json() == response2.json()

def test_bulk_minimize(authorized_client):
    """Test bulk minimization endpoint"""
    test_data = {
        "items": [
            {
                "automaton": {
                    "states": ["q0", "q1", "q2"],
                    "alphabet": ["a", "b"],
                    "transitions": {
                        "q0": {"a": "q1", "b": "q2"},
                        "q1": {"a": "q1", "b": "q2"},
                        "q2": {"a": "q2", "b": "q2"}
                    },
                    "start_state": "q0",
                    "accept_states": ["q2"]
                }
            }
        ],
        "parallel": True
    }
    
    response = authorized_client.post("/api/bulk/minimize", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["success_count"] > 0

def test_metrics_collection(client):
    """Test that metrics are being collected"""
    # Make a controlled number of requests
    for _ in range(3):
        client.get("/health")
    
    # Check metrics endpoint
    response = client.get("/metrics")
    assert response.status_code == 200
    metrics = response.content.decode()
    
    # Verify key metrics are present
    assert "http_requests_total" in metrics
    assert "http_request_duration_seconds" in metrics
    assert "active_connections" in metrics