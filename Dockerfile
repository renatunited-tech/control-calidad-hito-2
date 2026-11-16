# Segundo servicio del mismo repositorio GitHub; Streamlit usa app.py por separado.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY api/requirements_api.txt ./requirements_api.txt
RUN pip install --no-cache-dir -r requirements_api.txt
COPY api/ ./api/
COPY modelos/efficientnet_b0_best.pth ./modelos/efficientnet_b0_best.pth
EXPOSE 8080
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
