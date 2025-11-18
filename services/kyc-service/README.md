# KYC Service

Know Your Customer service with OCR and Face Recognition capabilities.

## Features

- **ID Card OCR**: Extract text from Ethiopian National ID cards using Tesseract
- **Face Recognition**: Detect, align, and extract 512-D embeddings from selfie photos using InsightFace
- **LLM Processing**: Extract structured data from OCR text using GitHub Models API

## Prerequisites

- Docker (for containerized deployment)
- Python 3.10+ (for local development)
- Environment variables:
  - `GITHUB_TOKEN`: Required - GitHub token for models API
  - `MODEL`: Optional - Model name (default: "openai/gpt-4.1")

## Setup

### Using Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   cd services/kyc-service
   docker build -t kyc-service .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     -p 8000:8000 \
     -e GITHUB_TOKEN=your_github_token_here \
     -e MODEL=openai/gpt-4.1 \
     --name kyc-service \
     kyc-service
   ```

3. **Access the service:**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Health: http://localhost:8000/health

### Local Development

1. **Install system dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y \
     tesseract-ocr \
     libtesseract-dev \
     libleptonica-dev \
     libgl1 \
     libglib2.0-0
   ```

2. **Create virtual environment:**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables:**
   ```bash
   export GITHUB_TOKEN=your_github_token_here
   export MODEL=openai/gpt-4.1  # Optional
   ```

5. **Run the service:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints

### POST `/kyc`

Complete KYC processing endpoint.

**Parameters:**
- `id_card` (file): ID card image (required)
- `selfie` (file): Selfie photo (required)
- `preprocess` (bool): Apply image preprocessing (default: true)
- `advanced` (bool): Use advanced OCR (default: false)

**Response:**
```json
{
  "status": "success",
  "id_card_data": {
    "status": "success",
    "data": {
      "document_type": "Ethiopian Digital ID Card",
      "full_name_english": "...",
      "date_of_birth_gc": "...",
      "expiry_date_gc": "...",
      "gender": "...",
      "citizenship": "ET",
      "fcn_number": "...",
      "extraction_quality": 7
    }
  },
  "selfie_processing": {
    "embeddings": [0.123, -0.456, ...],
    "embedding_dim": 512,
    "aligned": true
  }
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "KYC Service"
}
```

### GET `/`

Root endpoint with service information.

## Testing

### Using curl:

```bash
curl -X POST "http://localhost:8000/kyc?preprocess=true&advanced=false" \
  -F "id_card=@/path/to/id_card.jpg" \
  -F "selfie=@/path/to/selfie.jpg"
```

### Using Python:

```python
import requests

url = "http://localhost:8000/kyc"
files = {
    'id_card': open('id_card.jpg', 'rb'),
    'selfie': open('selfie.jpg', 'rb')
}
params = {
    'preprocess': True,
    'advanced': False
}

response = requests.post(url, files=files, params=params)
print(response.json())
```

## Troubleshooting

### Issue: "GITHUB_TOKEN environment variable is required"
**Solution:** Set the `GITHUB_TOKEN` environment variable before running the service.

### Issue: "No faces detected in image"
**Solution:** Ensure the selfie image contains a clear, front-facing face.

### Issue: InsightFace model download fails
**Solution:** The model will be automatically downloaded on first use. Ensure internet connectivity.

### Issue: Tesseract not found
**Solution:** Install Tesseract OCR system package:
```bash
sudo apt-get install tesseract-ocr
```

## Notes

- The InsightFace `buffalo_l` model will be automatically downloaded on first use (~500MB)
- Face embeddings are 512-dimensional vectors suitable for face comparison
- The service processes images in memory and doesn't store them
- For production, consider adding authentication, rate limiting, and logging

