# Flight Delay Prediction Service

MLOps HW2 - CI/CD Pipeline Implementation

**Student:** Efe Çetin - 220901578

## Project Structure

```
hw2/
├── .github/workflows/ci-cd.yml  # GitHub Actions pipeline
├── src/
│   ├── feature_engineering.py   # Hashed features & delay categorization
│   ├── model.py                 # Model loading and inference
│   └── api.py                   # Flask REST API
├── tests/
│   ├── test_feature_engineering.py  # Unit tests
│   └── test_integration.py          # Integration tests
├── model/                       # Trained model files
├── Dockerfile
├── requirements.txt
├── smoke_test.py
└── setup.cfg                    # Linting config
```

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run unit tests
python -m pytest tests/test_feature_engineering.py -v

# Run linting
flake8 src/

# Start API server
python -m src.api

# Run smoke tests (in another terminal)
python smoke_test.py
```

### Docker

```bash
# Build image
docker build -t flight-delay-api .

# Run container
docker run -p 8080:8080 flight-delay-api

# Test API
curl http://localhost:8080/health
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Predict delay category |
| `/features` | POST | Extract hashed features |

### Example Request

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"origin": "JFK", "dest": "LAX", "airline": "UA"}'
```

## CI/CD Pipeline

The GitHub Actions pipeline includes:

1. **Commit Stage**: Linting (Flake8) + Unit Tests
2. **Acceptance Stage**: Integration Tests + Docker Build + Smoke Tests
