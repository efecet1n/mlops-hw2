# Feature Engineering Module for Flight Delay Prediction
# MLOps HW2 - Efe Çetin

import hashlib
from typing import Optional


def hash_airport_code(code: str, num_buckets: int = 100) -> int:
    """
    Hash airport code into a bucket index using MD5 hashing.
    
    Args:
        code: Airport code (e.g., 'JFK', 'LAX')
        num_buckets: Number of buckets to hash into
    
    Returns:
        Bucket index (0 to num_buckets-1)
    """
    if not code or not isinstance(code, str):
        return 0
    encoded = code.encode("utf-8")
    hashed = hashlib.md5(encoded).hexdigest()
    hashed_int = int(hashed, 16)
    return hashed_int % num_buckets


def hash_airline_code(code: str, num_buckets: int = 20) -> int:
    """
    Hash airline code into a bucket index.
    
    Args:
        code: Airline code (e.g., 'UA', 'DL')
        num_buckets: Number of buckets to hash into
    
    Returns:
        Bucket index (0 to num_buckets-1)
    """
    if not code or not isinstance(code, str):
        return 0
    encoded = code.encode("utf-8")
    hashed = hashlib.md5(encoded).hexdigest()
    hashed_int = int(hashed, 16)
    return hashed_int % num_buckets


def categorize_delay(delay_minutes: Optional[float]) -> int:
    """
    Categorize delay into buckets for classification.
    
    Categories:
        0: On-time or small delay (0-10 minutes)
        1: Medium delay (11-30 minutes)
        2: Large delay (31+ minutes)
    
    Args:
        delay_minutes: Delay in minutes (can be negative for early arrivals)
    
    Returns:
        Category label (0, 1, or 2)
    """
    if delay_minutes is None:
        return 0
    if delay_minutes <= 10:
        return 0
    elif delay_minutes <= 30:
        return 1
    else:
        return 2


def extract_features(origin: str, dest: str, airline: str) -> dict:
    """
    Extract hashed features from flight data.
    
    Args:
        origin: Origin airport code
        dest: Destination airport code
        airline: Airline code
    
    Returns:
        Dictionary with hashed features
    """
    return {
        'origin_hash': hash_airport_code(origin, 100),
        'dest_hash': hash_airport_code(dest, 100),
        'airline_hash': hash_airline_code(airline, 20)
    }
