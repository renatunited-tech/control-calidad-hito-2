from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import timm
import torch
from PIL import Image, UnidentifiedImageError

# Modelo entrenado en el notebook. No se descargan pesos adicionales.
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "modelos/efficientnet_b0_best.pth"))
LABELS = ("correcta", "defectuosa")
_MODEL = None
_PREPROCESS = None
_LOCK = threading.Lock()


def load_model():
    global _MODEL, _PREPROCESS
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
                if (checkpoint.get("timm_model") != "efficientnet_b0" or
                        tuple(checkpoint.get("labels", [])) != LABELS):
                    raise RuntimeError("El checkpoint no coincide con el modelo esperado")
                model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                model.eval()
                # Igual preprocesamiento que en validación/prueba del notebook.
                cfg = timm.data.resolve_model_data_config(model)
                _PREPROCESS = timm.data.create_transform(**cfg)
                _MODEL = model
    return _MODEL, _PREPROCESS


def predict_image(raw: bytes) -> dict:
    if not raw:
        raise ValueError("Imagen vacía")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 20_000_000:
                raise ValueError("Imagen demasiado grande")
            rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Archivo de imagen inválido") from exc

    model, preprocess = load_model()
    x = preprocess(rgb).unsqueeze(0)
    with torch.inference_mode():
        scores = torch.softmax(model(x), dim=1)[0].tolist()
    predicted = int(scores[1] > scores[0])
    return {
        "modelo": "EfficientNet-B0",
        "clase_predicha": LABELS[predicted],
        "score_defectuosa": round(float(scores[1]), 6),
        "score_clase_predicha": round(float(scores[predicted]), 6),
        "puntajes_calibrados": False,
        "origen_entrenamiento": "dataset externo de piezas fundidas",
        "vinculo_lineas_abc": "no verificado",
    }
