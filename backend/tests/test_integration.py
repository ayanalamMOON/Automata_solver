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

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_detailed_health_check():
    """Test the detailed health check endpoint"""
    response = client.get("/health/detailed")
    assert response.status_code == 200
    data = response.json()
    assert "components" in data
    assert "redis" in data["components"]
    assert "system" in data["components"]

def test_metrics_endpoint():
    """Test the Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"http_requests_total" in response.content

def test_dfa_conversion():
    """Test DFA conversion endpoint"""
    test_data = {
        "regex": "a*b"
    }
    response = client.post("/convert", json=test_data)
    assert response.status_code == 200
    assert "dfa_svg" in response.json()

def test_batch_processing():
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
    response = client.post("/api/bulk/convert", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["success_count"] > 0

def test_automata_analysis():
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
    
    response = client.post("/api/analyze", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "deterministic" in data

def test_error_handling():
    """Test error handling for invalid input"""
    test_data = {
        "regex": "("  # Invalid regex
    }
    response = client.post("/convert", json=test_data)
    assert response.status_code == 400
    assert "error" in response.json()

def test_rate_limiting():
    """Test rate limiting middleware"""
    # Make multiple requests quickly
    responses = [
        client.get("/health")
        for _ in range(int(os.getenv('RATE_LIMIT_PER_MINUTE', 60)) + 1)
    ]
    
    # At least one request should be rate limited
    assert any(r.status_code == 429 for r in responses)

def test_caching():
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

def test_security():
    """Test security endpoints and authentication"""
    # Try accessing admin endpoint without authentication
    response = client.get("/api/admin/stats")
    assert response.status_code == 401  # Unauthorized

    # Test with invalid token
    response = client.get(
        "/api/admin/stats",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401

def test_bulk_minimize():
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
    
    response = client.post("/api/bulk/minimize", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["success_count"] > 0

def test_metrics_collection():
    """Test that metrics are being collected"""
    # Make some requests to generate metrics
    client.get("/health")
    client.get("/metrics")
    
    # Check metrics endpoint
    response = client.get("/metrics")
    assert response.status_code == 200
    metrics = response.content.decode()
    
    # Verify key metrics are present
    assert "http_requests_total" in metrics
    assert "http_request_duration_seconds" in metrics
    assert "active_connections" in metrics