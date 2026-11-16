# Importar librerías y preparar la interfaz
import json
from pathlib import Path
import pandas as pd
import streamlit as st
import timm
import torch
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title="Control de calidad | Inspección visual", layout="wide")

# Rutas de los archivos generados durante el entrenamiento
BASE = Path(__file__).resolve().parent
MODELO_PATH = BASE / "modelos" / "efficientnet_b0_best.pth"
METRICAS_PATH = BASE / "resultados" / "metricas_prueba.csv"
SELECCION_PATH = BASE / "resultados" / "seleccion_modelo.json"
CLASES = ("correcta", "defectuosa")
AZUL = "#197A96"

# Paleta industrial: azul petróleo, gris acero y acentos ámbar
st.markdown("""
<style>
    .stApp { background-color: #F3F6FA; color: #15283D; }
    .block-container { max-width: 1150px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero {
        background: linear-gradient(115deg, #10283F 0%, #18506C 59%, #176E87 100%);
        border: 1px solid #2D6580; border-radius: 20px; padding: 35px 40px;
        margin-bottom: 25px; box-shadow: 0 7px 24px rgba(16,40,63,.10);
    }
    .hero h1 { color: #FFFFFF; font-size: 2.55rem; margin: 12px 0; }
    .hero p { color: #D7E8F0; font-size: 1.05rem; margin-bottom: 0; }
    .etiqueta { color: #A8DCE8; font-size: .78rem; font-weight: 700; letter-spacing: 2px; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF; border: 1px solid #D9E3EB;
        border-radius: 15px; padding: 20px;
    }
    div[data-testid="stImage"] img { border-radius: 14px; }
    div[data-testid="stFileUploader"] { background-color: #FFFFFF; border-radius: 12px; }
    h2, h3 { color: #173B54; }
    .subtle { font-size: .91rem; color: #52697D; }
    .stButton > button[kind="primary"] { background: #176E87; border-color: #176E87; }
</style>
""", unsafe_allow_html=True)

# Encabezado principal (misma estructura del Certamen 1)
st.markdown("""
<div class="hero">
    <div class="etiqueta">CONTROL INTELIGENTE DE CALIDAD</div>
    <h1>Inspección visual de piezas</h1>
    <p>Clasificación de fotografías industriales con EfficientNet-B0 y evaluación del desempeño del modelo.</p>
</div>
""", unsafe_allow_html=True)

# Dos pestañas: inspección y documentación del modelo
tab1, tab2 = st.tabs(["Inspeccionar pieza", "Sobre el modelo"])

with tab1:
    st.subheader("Analiza una pieza")
    st.write("Sube una fotografía para obtener una clasificación preliminar de su estado visual.")
    archivo = st.file_uploader("Selecciona una fotografía", type=["jpg", "jpeg", "png"])
    imagen = None
    if archivo is not None:
        try:
            imagen = Image.open(archivo).convert("RGB")
            if imagen.width * imagen.height > 20_000_000:
                st.error("La fotografía supera el tamaño admitido.")
                imagen = None
        except (OSError, UnidentifiedImageError):
            st.error("No se pudo leer la fotografía. Selecciona un archivo JPG o PNG válido.")
    if imagen is not None:
        col_imagen, col_resultado = st.columns([1, 1.2], gap="large")
        with col_imagen:
            st.image(imagen, caption="Fotografía cargada", use_container_width=True)
# Reconstruir EfficientNet con el checkpoint que ganó en validación
@st.cache_resource(show_spinner="Cargando EfficientNet-B0...")
def cargar_modelo(ruta: str):
    checkpoint = torch.load(ruta, map_location="cpu", weights_only=True)
    if checkpoint.get("timm_model") != "efficientnet_b0":
        raise ValueError("El checkpoint no corresponde a EfficientNet-B0.")
    if tuple(checkpoint.get("labels", ())) != CLASES:
        raise ValueError("Las clases del checkpoint no coinciden con las esperadas.")
    modelo = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    modelo.load_state_dict(checkpoint["state_dict"], strict=True)
    modelo.eval()
    # Mismo preprocesamiento utilizado en validación y prueba del notebook de entrenamiento
    config = timm.data.resolve_model_data_config(modelo)
    preprocesamiento = timm.data.create_transform(**config)
    return modelo, preprocesamiento

# Clasificar la imagen sin volver a entrenar ni descargar pesos preentrenados
@torch.inference_mode()
def clasificar_imagen(imagen_rgb):
    modelo, preprocesamiento = cargar_modelo(str(MODELO_PATH))
    x = preprocesamiento(imagen_rgb).unsqueeze(0)
    puntajes = torch.softmax(modelo(x), dim=1)[0].tolist()
    indice = 0 if puntajes[0] >= puntajes[1] else 1
    return {"clase": CLASES[indice], "score": float(puntajes[indice]), "scores": puntajes}

# Mostrar los resultados al lado de la imagen, igual que en el Certamen 1
prediccion = None
if imagen is not None:
    with col_resultado:
        st.subheader("Resultado de la clasificación")
        if not MODELO_PATH.is_file():
            st.error("Faltan los pesos del modelo en modelos/efficientnet_b0_best.pth.")
        else:
            try:
                prediccion = clasificar_imagen(imagen)
                st.metric("Estado visual estimado", prediccion["clase"].capitalize())
                st.metric("Puntuación de la clase", f"{prediccion['score']:.1%}")
            except Exception as exc:
                st.error(f"No se pudo ejecutar EfficientNet: {exc}")
# Mostrar los puntajes para ambas clases
with tab1:
    if prediccion is not None:
        tabla = pd.DataFrame({
            "Estado": ["Correcta", "Defectuosa"],
            "Puntuación (%)": [round(v * 100, 2) for v in prediccion["scores"]],
        })
        st.subheader("Puntuación por categoría")
        st.bar_chart(tabla, x="Estado", y="Puntuación (%)", color=AZUL,
                     horizontal=True, height=240, sort=False)
        with st.expander("Ver valores exactos"):
            st.dataframe(tabla, hide_index=True, use_container_width=True)

        # Los valores softmax no deben interpretarse como probabilidades calibradas
        if prediccion["score"] < 0.70:
            st.warning("La puntuación del modelo es poco concluyente; solicita revisión visual humana.")
        if prediccion["clase"] == "defectuosa":
            st.warning("Posible defecto visual. La pieza requiere inspección humana antes de decidir su destino.")
        else:
            st.info("No se identificó un defecto según el modelo. Esto no sustituye la inspección exigida por el proceso.")
        st.caption("Los puntajes no están calibrados como probabilidades. El modelo no identifica causas raíz ni justifica una detención automática de producción.")
# Mantener la recomendación opcional de Gemini con el enfoque de calidad industrial
with tab1:
    if prediccion is not None:
        st.divider()
        st.subheader("Orientación para la inspección")
        st.write("Puedes solicitar una orientación textual basada en la clasificación de la fotografía.")
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
            modelo_gemini = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
        except Exception:
            api_key = ""
            modelo_gemini = "gemini-2.5-flash"

        if not api_key:
            st.caption("La consulta a Gemini es opcional. Para habilitarla, configura GEMINI_API_KEY en los secretos de Streamlit.")
        if st.button("Obtener recomendación de inspección", type="primary", disabled=not bool(api_key)):
            # Gemini interpreta la predicción, pero no reemplaza los modelos estadísticos ni la revisión humana
            prompt = f"""
            Actúa como asistente de apoyo para inspección visual industrial.
            Una CNN EfficientNet-B0 clasificó una fotografía de una pieza de fundición como
            '{prediccion['clase']}' con puntuación softmax {prediccion['score']:.2%}.
            El score NO es una probabilidad calibrada. Esta fotografía pertenece a un
            dataset académico externo, sin vínculo demostrado con las líneas A/B/C.
            En español, entrega un diagnóstico PRELIMINAR y hasta tres recomendaciones
            prácticas de verificación visual o documental. No afirmes conocer la causa
            raíz, severidad, costos, tolerancias o procesos reales. No ordenes detener
            máquinas ni rechazar piezas sin aprobación humana.
            """
            try:
                from google import genai
                with st.spinner("Generando orientación..."):
                    cliente = genai.Client(api_key=api_key)
                    respuesta = cliente.models.generate_content(model=modelo_gemini, contents=prompt)
                if respuesta.text:
                    st.write(respuesta.text)
                else:
                    st.warning("Gemini no entregó una recomendación.")
            except Exception as exc:
                # Diagnóstico seguro: no mostrar la clave ni datos sensibles de la solicitud.
                import re
                detalle = str(exc).replace(str(api_key), "[CLAVE OCULTA]")
                detalle = re.sub(r"AIza[0-9A-Za-z_-]{16,}", "[CLAVE OCULTA]", detalle)
                detalle = re.sub(r"(?i)(api[_ -]?key[=: ]+)[^,\s}]+", r"\1[OCULTO]", detalle)
                codigo = getattr(exc, "code", None)
                if codigo is None:
                    match = re.search(r"\b(400|401|403|404|429|500|503)\b", detalle)
                    codigo = match.group(1) if match else None
                st.error("No se pudo completar la consulta a Gemini.")
                if str(codigo) in ("401", "403"):
                    st.info("Error de autorización. Comprueba que la clave corresponde a la API de Gemini y está habilitada para este proyecto.")
                elif str(codigo) == "404":
                    st.info("El modelo configurado no se encontró o no está disponible para esta clave. Revisa GEMINI_MODEL.")
                elif str(codigo) == "429":
                    st.info("Se alcanzó el límite de solicitudes o de cuota del proyecto de Gemini.")
                elif isinstance(exc, (ImportError, ModuleNotFoundError)):
                    st.info("Falta instalar google-genai en las dependencias de Streamlit Cloud (requirements.txt).")
                with st.expander("Detalle técnico del error (sin credenciales)"):
                    st.code(f"{type(exc).__name__} (código {codigo or 'no identificado'}): {detalle[:1100]}", language="text")
# Cargar los resultados reales generados por el notebook de comparación
@st.cache_data
def leer_resultados():
    seleccion = None
    metricas = None
    if SELECCION_PATH.is_file():
        with open(SELECCION_PATH, "r", encoding="utf8") as f:
            seleccion = json.load(f)
    if METRICAS_PATH.is_file():
        metricas = pd.read_csv(METRICAS_PATH)
    return seleccion, metricas

with tab2:
    st.subheader("Dataset de inspección industrial")
    st.write(
        "Fotografías de piezas fundidas de la colección Casting Product Image Data for Quality Inspection. "
        "Se utilizaron las imágenes originales, con etiquetas 'correcta' y 'defectuosa'."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Fotografías originales", "1.300")
    c2.metric("Entrenamiento", "928")
    c3.metric("Validación / Prueba", "186 / 186")
    st.caption("Separación por grupos de similitud perceptual para reducir duplicados entre conjuntos.")
    st.markdown(
        "**Fuente:** [Casting Product Image Data for Quality Inspection (Kaggle)]"
        "(https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)"
    )
    st.divider()

    # Distribución real de las dos categorías del dataset
    st.subheader("Distribución de las categorías")
    distribucion = pd.DataFrame({
        "Clase": ["Correcta", "Defectuosa"],
        "Total": [519, 781],
        "Entrenamiento": [371, 557],
        "Validación": [74, 112],
        "Prueba": [74, 112],
    })
    st.bar_chart(distribucion, x="Clase", y="Total", color=AZUL,
                 horizontal=True, height=230, sort=False)
    with st.expander("Ver distribución de entrenamiento, validación y prueba"):
        st.dataframe(distribucion, hide_index=True, use_container_width=True)

    # Ejemplos visuales guardados por el notebook de la aplicación
    st.markdown("**Ejemplos de ambas categorías**")
    ejemplos = [("correcta", "Pieza correcta"), ("defectuosa", "Pieza defectuosa")]
    columnas = st.columns(2)
    for i, (nombre, titulo) in enumerate(ejemplos):
        with columnas[i]:
            ruta = BASE / "ejemplos" / f"{nombre}.jpg"
            if ruta.is_file():
                st.image(str(ruta), caption=titulo, use_container_width=True)
            else:
                st.caption(f"Ejemplo no disponible: {titulo.lower()}.")
    st.divider()

    # Comparar modelos con métricas calculadas, sin valores de relleno
    st.subheader("Comparación de modelos")
    seleccion, metricas = leer_resultados()
    if metricas is None or metricas.empty:
        st.info("Todavía no se encuentra resultados/metricas_prueba.csv.")
    else:
        columnas_resultados = ["modelo", "accuracy_balanceada", "accuracy", "f1_defectuosa", "recall_defectuosa", "fp", "fn"]
        disponibles = [c for c in columnas_resultados if c in metricas.columns]
        vista = metricas[disponibles].copy()
        nombres = {"modelo": "Modelo", "accuracy_balanceada": "Accuracy balanceada",
                   "accuracy": "Accuracy", "f1_defectuosa": "F1 defectuosas",
                   "recall_defectuosa": "Recall defectuosas", "fp": "FP", "fn": "FN"}
        vista = vista.rename(columns=nombres)
        st.dataframe(vista, hide_index=True, use_container_width=True,
                     column_config={
                         "Accuracy balanceada": st.column_config.NumberColumn(format="%.2f"),
                         "Accuracy": st.column_config.NumberColumn(format="%.2f"),
                         "F1 defectuosas": st.column_config.NumberColumn(format="%.2f"),
                         "Recall defectuosas": st.column_config.NumberColumn(format="%.2f"),
                     })
        if "accuracy_balanceada" in metricas.columns:
            st.bar_chart(metricas, x="modelo", y="accuracy_balanceada", color=AZUL,
                         horizontal=True, height=220, sort=False)
            st.caption("Accuracy balanceada expresada como proporción entre 0 y 1. Valores del conjunto de prueba.")

    if seleccion is not None:
        nombre = seleccion.get("modelo_seleccionado", "No registrado")
        st.success(f"Modelo seleccionado en validación: {nombre}.")
        st.caption("La elección se hizo en validación, no utilizando los resultados de prueba.")
        st.divider()

        # Métricas y matriz de confusión del modelo seleccionado
        st.subheader("Desempeño del modelo seleccionado")
        m = seleccion.get("metrica_prueba", {})
        p1, p2, p3 = st.columns(3)
        p1.metric("Accuracy en prueba", f"{m.get('accuracy', 0):.1%}")
        p2.metric("Accuracy balanceada", f"{m.get('accuracy_balanceada', 0):.1%}")
        p3.metric("F1 defectuosas", f"{m.get('f1_defectuosa', 0):.1%}")
        st.caption(f"Prueba: {m.get('n', '—')} fotografías · FP: {m.get('fp', '—')} · FN: {m.get('fn', '—')}.")
        matriz = m.get("matriz_confusion")
        if matriz:
            with st.expander("Ver matriz de confusión"):
                tabla_matriz = pd.DataFrame(matriz,
                                           index=["Real correcta", "Real defectuosa"],
                                           columns=["Pred. correcta", "Pred. defectuosa"])
                st.dataframe(tabla_matriz, use_container_width=True)
                st.caption("Filas: clase real. Columnas: clase predicha.")
    else:
        st.info("Todavía no se encuentra resultados/seleccion_modelo.json.")

    st.divider()
    st.subheader("Alcance y limitaciones")
    st.info(
        "Este modelo fue evaluado sobre un dataset académico de piezas fundidas. "
        "No ha sido validado en una planta real ni existe correspondencia demostrada "
        "entre estas imágenes y las líneas A, B y C del sistema de n8n. "
        "Su resultado es una alerta preliminar sujeta a inspección humana."
    )
