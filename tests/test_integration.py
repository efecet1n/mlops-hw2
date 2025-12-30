# Integration/Component Tests
# MLOps HW2 - Efe Çetin

import unittest
import os
import sys
import json

# Add project root to path for CI compatibility
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.api import app


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for the Flask API."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client."""
        cls.client = app.test_client()
        cls.client.testing = True
    
    def test_health_endpoint(self):
        """Test /health endpoint returns 200."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_predict_endpoint_success(self):
        """Test /predict endpoint with valid data."""
        payload = {
            "origin": "JFK",
            "dest": "LAX",
            "airline": "UA"
        }
        
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('origin_hash', data)
        self.assertIn('dest_hash', data)
        self.assertIn('airline_hash', data)
        self.assertIn('prediction', data)
        self.assertIn(data['prediction'], [0, 1, 2])
    
    def test_predict_endpoint_missing_field(self):
        """Test /predict endpoint with missing required field."""
        payload = {
            "origin": "JFK",
            # Missing 'dest' and 'airline'
        }
        
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_predict_endpoint_no_json(self):
        """Test /predict endpoint with no JSON data."""
        response = self.client.post('/predict')
        self.assertEqual(response.status_code, 400)
    
    def test_features_endpoint_success(self):
        """Test /features endpoint with valid data."""
        payload = {
            "origin": "SFO",
            "dest": "ORD",
            "airline": "DL"
        }
        
        response = self.client.post(
            '/features',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('origin_hash', data)
        self.assertIn('dest_hash', data)
        self.assertIn('airline_hash', data)
        
        # Verify hashes are within valid range
        self.assertGreaterEqual(data['origin_hash'], 0)
        self.assertLess(data['origin_hash'], 100)
    
    def test_features_endpoint_consistency(self):
        """Test that same input produces same features."""
        payload = {"origin": "JFK", "dest": "LAX", "airline": "UA"}
        
        response1 = self.client.post(
            '/features',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        response2 = self.client.post(
            '/features',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        data1 = json.loads(response1.data)
        data2 = json.loads(response2.data)
        
        self.assertEqual(data1, data2)


if __name__ == '__main__':
    unittest.main()
