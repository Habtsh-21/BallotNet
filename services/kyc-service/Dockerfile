# -----------------------------
# Base image
# -----------------------------
    FROM python:3.10-slim

    # Avoid interactive prompts during apt installs
    ENV DEBIAN_FRONTEND=noninteractive
    ENV DEBCONF_NOWARNINGS=yes
    
    # Set working directory
    WORKDIR /app
    
    # -----------------------------
    # 1️⃣ Runtime dependencies (rarely change)
    # -----------------------------
    RUN apt-get update && apt-get install -y --no-install-recommends \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender1 \
            libgomp1 \
            libgl1 \
            tesseract-ocr \
            libtesseract-dev \
            libleptonica-dev \
        && rm -rf /var/lib/apt/lists/*
    
    # -----------------------------
    # 2️⃣ Build dependencies (for compiling packages like insightface)
    # -----------------------------
    RUN apt-get update && apt-get install -y --no-install-recommends \
            build-essential \
            g++ \
            cmake \
            git \
            pkg-config \
            libjpeg-dev \
            libpng-dev \
            zlib1g-dev \
        && rm -rf /var/lib/apt/lists/*
    
    # -----------------------------
    # 3️⃣ Python dependencies
    # -----------------------------
    # Copy only requirements first to leverage Docker cache
    COPY requirements.txt .
    
    # Upgrade pip/setuptools/wheel first for smooth installs
    RUN pip install --no-cache-dir -r requirements.txt
    
    # -----------------------------
    # 4️⃣ Optional: Pre-load InsightFace model
    # -----------------------------
    RUN python -c "from insightface.app import FaceAnalysis; \
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
        app.prepare(ctx_id=-1, det_size=(640,640))" || true
    
    # -----------------------------
    # 5️⃣ Copy application code
    # -----------------------------
    COPY app ./app
    
    # -----------------------------
    # 6️⃣ Expose port and CMD
    # -----------------------------
    EXPOSE 8000
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    