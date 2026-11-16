from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from api.model_service import MODEL_PATH, predict_image

app = FastAPI(title="Hito 2 - Inspección visual", version="1.0.0")
MAX_SIZE = 8 * 1024 * 1024


def authorize(x_api_token: str | None):
    # La clave se configura exclusivamente como variable secreta del hosting.
    expected = os.environ.get("INFERENCE_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Token de servicio no configurado")
    if not x_api_token or not hmac.compare_digest(x_api_token, expected):
        raise HTTPException(status_code=401, detail="No autorizado")


@app.get("/health")
def health():
    status = "ok" if Path(MODEL_PATH).is_file() else "model_missing"
    return {"status": status, "modelo": "EfficientNet-B0"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_token: str | None = Header(default=None)):
    authorize(x_api_token)
    try:
        raw = await file.read(MAX_SIZE + 1)
        if len(raw) > MAX_SIZE:
            raise HTTPException(status_code=413, detail="Imagen supera 8 MB")
        result = predict_image(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Modelo no disponible") from exc
    finally:
        await file.close()
    return {"imagen": Path(file.filename or "imagen").name, **result}
