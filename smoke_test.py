# Smoke Test for Flight Delay Prediction API
# MLOps HW2 - Efe Çetin

import requests
import sys
import time

BASE_URL = "http://localhost:8080"


def test_health():
    """Test health check endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    
    if response.status_code != 200:
        raise AssertionError(f"Health check failed: {response.status_code}")
    
    data = response.json()
    if data.get("status") != "healthy":
        raise AssertionError(f"Unexpected health status: {data}")
    
    print("[PASS] Health check passed")
    return True


def test_predict():
    """Test prediction endpoint."""
    print("Testing /predict endpoint...")
    
    payload = {
        "origin": "JFK",
        "dest": "LAX",
        "airline": "UA"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        timeout=10
    )
    
    if response.status_code != 200:
        raise AssertionError(f"Predict failed: {response.status_code} - {response.text}")
    
    data = response.json()
    
    # Verify response structure
    required_fields = ["origin_hash", "dest_hash", "airline_hash", "prediction"]
    for field in required_fields:
        if field not in data:
            raise AssertionError(f"Missing field in response: {field}")
    
    # Verify prediction is valid category
    if data["prediction"] not in [0, 1, 2]:
        raise AssertionError(f"Invalid prediction: {data['prediction']}")
    
    print(f"[PASS] Prediction endpoint passed (prediction={data['prediction']})")
    return True


def test_features():
    """Test feature extraction endpoint."""
    print("Testing /features endpoint...")
    
    payload = {
        "origin": "SFO",
        "dest": "ORD",
        "airline": "DL"
    }
    
    response = requests.post(
        f"{BASE_URL}/features",
        json=payload,
        timeout=10
    )
    
    if response.status_code != 200:
        raise AssertionError(f"Features failed: {response.status_code}")
    
    data = response.json()
    
    # Verify all hashes are present and valid
    for key in ["origin_hash", "dest_hash", "airline_hash"]:
        if key not in data:
            raise AssertionError(f"Missing {key} in response")
        if not isinstance(data[key], int):
            raise AssertionError(f"{key} is not an integer")
    
    print("[PASS] Features endpoint passed")
    return True


def run_smoke_tests():
    """Run all smoke tests."""
    print("=" * 50)
    print("Starting Smoke Tests")
    print("=" * 50)
    print()
    
    # Wait for service to be ready
    max_retries = 5
    for i in range(max_retries):
        try:
            requests.get(f"{BASE_URL}/health", timeout=5)
            break
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                print(f"Waiting for service... ({i+1}/{max_retries})")
                time.sleep(2)
            else:
                print("[FAIL] Service not available")
                sys.exit(1)
    
    # Run tests
    tests = [test_health, test_predict, test_features]
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__} error: {e}")
            failed += 1
    
    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    if failed > 0:
        print("\n[FAIL] Smoke tests FAILED")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All smoke tests PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    run_smoke_tests()
