# Unit Tests for Feature Engineering
# MLOps HW2 - Efe Çetin

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.feature_engineering import (
    hash_airport_code,
    hash_airline_code,
    categorize_delay,
    extract_features
)


class TestHashAirportCode(unittest.TestCase):
    """Test cases for airport code hashing."""
    
    def test_returns_valid_bucket_range(self):
        """Hash should return value between 0 and num_buckets-1."""
        result = hash_airport_code("JFK", num_buckets=100)
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, 100)
    
    def test_is_deterministic(self):
        """Same input should always produce same output."""
        result1 = hash_airport_code("LAX")
        result2 = hash_airport_code("LAX")
        self.assertEqual(result1, result2)
    
    def test_different_inputs_can_differ(self):
        """Different inputs may produce different outputs."""
        result_jfk = hash_airport_code("JFK")
        result_lax = hash_airport_code("LAX")
        # They might collide but usually won't
        # This test just ensures no errors occur
        self.assertIsInstance(result_jfk, int)
        self.assertIsInstance(result_lax, int)
    
    def test_handles_empty_string(self):
        """Empty string should return 0."""
        result = hash_airport_code("")
        self.assertEqual(result, 0)
    
    def test_handles_none(self):
        """None should return 0."""
        result = hash_airport_code(None)
        self.assertEqual(result, 0)
    
    def test_custom_bucket_size(self):
        """Should respect custom bucket size."""
        result = hash_airport_code("JFK", num_buckets=10)
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, 10)


class TestHashAirlineCode(unittest.TestCase):
    """Test cases for airline code hashing."""
    
    def test_returns_valid_bucket_range(self):
        """Hash should return value between 0 and num_buckets-1."""
        result = hash_airline_code("UA", num_buckets=20)
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, 20)
    
    def test_is_deterministic(self):
        """Same input should always produce same output."""
        result1 = hash_airline_code("DL")
        result2 = hash_airline_code("DL")
        self.assertEqual(result1, result2)


class TestCategorizeDelay(unittest.TestCase):
    """Test cases for delay categorization."""
    
    def test_on_time_zero_delay(self):
        """Zero delay should be category 0."""
        self.assertEqual(categorize_delay(0), 0)
    
    def test_on_time_small_delay(self):
        """Small delay (<=10 min) should be category 0."""
        self.assertEqual(categorize_delay(5), 0)
        self.assertEqual(categorize_delay(10), 0)
    
    def test_early_arrival(self):
        """Early arrival (negative) should be category 0."""
        self.assertEqual(categorize_delay(-15), 0)
    
    def test_medium_delay(self):
        """Medium delay (11-30 min) should be category 1."""
        self.assertEqual(categorize_delay(11), 1)
        self.assertEqual(categorize_delay(20), 1)
        self.assertEqual(categorize_delay(30), 1)
    
    def test_large_delay(self):
        """Large delay (>30 min) should be category 2."""
        self.assertEqual(categorize_delay(31), 2)
        self.assertEqual(categorize_delay(60), 2)
        self.assertEqual(categorize_delay(120), 2)
    
    def test_handles_none(self):
        """None should return 0."""
        self.assertEqual(categorize_delay(None), 0)


class TestExtractFeatures(unittest.TestCase):
    """Test cases for feature extraction."""
    
    def test_returns_all_features(self):
        """Should return dict with all required features."""
        result = extract_features("JFK", "LAX", "UA")
        self.assertIn('origin_hash', result)
        self.assertIn('dest_hash', result)
        self.assertIn('airline_hash', result)
    
    def test_features_are_integers(self):
        """All features should be integers."""
        result = extract_features("JFK", "LAX", "UA")
        self.assertIsInstance(result['origin_hash'], int)
        self.assertIsInstance(result['dest_hash'], int)
        self.assertIsInstance(result['airline_hash'], int)


if __name__ == '__main__':
    unittest.main()
