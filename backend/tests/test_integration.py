import pytest
from fastapi.testclient import TestClient
from main import app
import redis
import os
from dotenv import load_dotenv
from metrics import reset_metrics

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
def setup_test_env():
    """Set up test environment variables and reset metrics"""
    os.environ['TESTING'] = 'true'
    reset_metrics()
    redis_client.flushdb()
    yield
    os.environ.pop('TESTING', None)
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
    response = authorized_client.post("/api/analyze", json={
        "automaton": {
            "states": ["q0", "q1"],
            "alphabet": ["0", "1"],
            "transitions": {
                "q0": {"0": "q1", "1": "q0"},
                "q1": {"0": "q1", "1": "q0"}
            },
            "start_state": "q0",
            "accept_states": ["q1"]
        },
        "properties": ["deterministic", "minimal"]
    })
    assert response.status_code == 200
    result = response.json()
    assert "deterministic" in result
    assert "minimal" in result


def test_error_handling(client):
    """Test error handling for invalid input"""
    test_data = {
        "regex": "("  # Invalid regex
    }
    response = client.post("/convert", json=test_data)
    assert response.status_code == 400
    assert "error" in response.json() or "detail" in response.json()


def test_rate_limiting(client):
    """Test rate limiting with test mode bypassing"""
    # In test mode, rate limiting should be bypassed
    for _ in range(100):  # More than the rate limit
        response = client.get("/api/analyze/test")
        assert response.status_code != 429  # Should not hit rate limit


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
    """Test bulk minimization with known reducible automaton"""
    response = authorized_client.post("/api/bulk/minimize", json={
        "items": [{
            "automaton": {
                "states": ["q0", "q1", "q2"],  # q1 and q2 are equivalent
                "alphabet": ["0", "1"],
                "transitions": {
                    "q0": {"0": "q1", "1": "q0"},
                    "q1": {"0": "q2", "1": "q0"},
                    "q2": {"0": "q2", "1": "q0"}
                },
                "start_state": "q0",
                "accept_states": ["q1", "q2"]
            }
        }],
        "parallel": True
    })
    assert response.status_code == 200
    result = response.json()
    # Should have minimized at least one automaton
    assert result["minimized_count"] > 0


def test_metrics_collection(client):
    """Test metrics collection in test mode"""
    # Reset metrics before test
    reset_metrics()
    
    # Make some requests to generate metrics
    client.get("/health")
    client.get("/metrics")
    
    # Get metrics
    response = client.get("/metrics")
    assert response.status_code == 200
    metrics_text = response.text
    
    # Check for expected metrics
    assert 'http_requests_total' in metrics_text
    assert 'http_request_duration_seconds' in metrics_text
    assert 'active_connections' in metrics_text


def test_regex_validation():
    """Test regex validation endpoint"""
    response = client.post("/api/validate_regex", json={"regex": "a*b"})
    assert response.status_code == 200
    assert response.json()["is_valid"]

    response = client.post("/api/validate_regex", json={"regex": "a**b"})
    assert response.status_code == 200
    assert not response.json()["is_valid"]


def test_regex_to_dfa_conversion():
    """Test regex to DFA conversion endpoint"""
    response = client.post("/api/convert", json={"regex": "a*b"})
    assert response.status_code == 200
    data = response.json()
    
    # Verify DFA structure
    assert "states" in data
    assert "alphabet" in data
    assert "transitions" in data
    assert "start_state" in data
    assert "accept_states" in data
    assert "visualization_state" in data

    # Verify visualization state structure
    vis_state = data["visualization_state"]
    assert "nodes" in vis_state
    assert "edges" in vis_state
    
    # Verify the converted DFA accepts correct strings
    dfa_response = client.post("/api/simulate/step_by_step", json={
        "states": data["states"],
        "alphabet": data["alphabet"],
        "transitions": data["transitions"],
        "start_state": data["start_state"],
        "accept_states": data["accept_states"],
        "input_string": "aab"
    })
    assert dfa_response.status_code == 200
    assert dfa_response.json()["accepted"]


def test_regex_api_validation():
    """Test regex validation endpoint with various patterns"""
    # Valid patterns
    patterns = ['a', 'ab', 'a|b', 'a*', '(a|b)*', 'a+b?c*', '(a|b)*abb']
    for pattern in patterns:
        response = client.post("/api/validate_regex", json={"regex": pattern})
        assert response.status_code == 200
        assert response.json()["is_valid"]

    # Invalid patterns
    invalid_patterns = ['', '*', 'a**b', '(ab', 'ab)', 'a||b']
    for pattern in invalid_patterns:
        response = client.post("/api/validate_regex", json={"regex": pattern})
        assert response.status_code == 200
        assert not response.json()["is_valid"]


def test_regex_api_conversion():
    """Test regex to DFA conversion endpoint"""
    # Test basic conversion
    response = client.post("/api/convert", json={"regex": "a*b"})
    assert response.status_code == 200
    data = response.json()
    
    # Verify DFA structure
    assert "states" in data
    assert "alphabet" in data
    assert "transitions" in data
    assert "start_state" in data
    assert "accept_states" in data
    assert "visualization_state" in data
    
    # Verify visualization structure
    vis_state = data["visualization_state"]
    assert "nodes" in vis_state
    assert "edges" in vis_state
    
    # Test that resulting DFA accepts correct strings
    dfa_data = {
        "states": data["states"],
        "alphabet": data["alphabet"],
        "transitions": data["transitions"],
        "start_state": data["start_state"],
        "accept_states": data["accept_states"],
        "input_string": "aab"
    }
    
    response = client.post("/api/simulate/step_by_step", json=dfa_data)
    assert response.status_code == 200
    assert response.json()["accepted"]
    
    # Test error handling with invalid regex
    response = client.post("/api/convert", json={"regex": "a**b"})
    assert response.status_code == 400


def test_regex_step_by_step_simulation():
    """Test step-by-step simulation of converted regex DFA"""
    # First convert a regex to DFA
    regex = "(a|b)*abb"
    conv_response = client.post("/api/convert", json={"regex": regex})
    assert conv_response.status_code == 200
    dfa_data = conv_response.json()
    
    # Test cases that should be accepted
    accept_cases = ["abb", "aabb", "babb", "abababb"]
    for test_str in accept_cases:
        sim_data = {**dfa_data, "input_string": test_str}
        response = client.post("/api/simulate/step_by_step", json=sim_data)
        assert response.status_code == 200
        result = response.json()
        assert result["accepted"]
        assert len(result["steps"]) == len(test_str) + 1  # +1 for initial state
        
    # Test cases that should be rejected
    reject_cases = ["ab", "ba", "abba"]
    for test_str in reject_cases:
        sim_data = {**dfa_data, "input_string": test_str}
        response = client.post("/api/simulate/step_by_step", json=sim_data)
        assert response.status_code == 200
        result = response.json()
        assert not result["accepted"]
        
    # Test visualization state in steps
    test_str = "abb"
    sim_data = {**dfa_data, "input_string": test_str}
    response = client.post("/api/simulate/step_by_step", json=sim_data)
    result = response.json()
    
    # Check each step has proper visualization state
    for step in result["steps"]:
        assert "visualization_state" in step
        vis_state = step["visualization_state"]
        assert "nodes" in vis_state
        assert "edges" in vis_state
        # Verify exactly one node is active at each step
        assert sum(1 for node in vis_state["nodes"] if node["active"]) == 1