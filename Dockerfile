FROM python:3.11-slim

WORKDIR /app

# Install deps first (layer-cached until requirements change)
COPY pyproject.toml requirements.txt ./
COPY src/ src/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -e .

# Copy app code and runtime artifacts
COPY app/ app/
COPY models/model.joblib models/model.joblib
COPY reports/train_feature_stats.json reports/train_feature_stats.json

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
