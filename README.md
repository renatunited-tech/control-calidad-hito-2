# Control inteligente de calidad de producción

Proyecto desarrollado para el ramo IA Generativa para la Gestión Industrial. La idea es apoyar el monitoreo de calidad de una planta, combinando el análisis de registros de producción con la clasificación de fotografías de piezas industriales.

## Qué hicimos

El flujo principal se desarrolló en n8n. Lee los registros de Google Sheets, calcula tasas de defectos, alertas y pronósticos con Python, y consulta fotografías guardadas en Google Drive. Las imágenes se clasifican con EfficientNet-B0 mediante una API alojada en Railway. Luego, Gemini interpreta los resultados y el sistema envía un informe por Gmail y Telegram, además de guardar un historial de las ejecuciones.

También desarrollamos una aplicación en Streamlit para subir una fotografía, ver su clasificación y solicitar una orientación de inspección con Gemini.

Aplicación: https://control-calidad-hito-2.streamlit.app/

## Archivos del repositorio

- `app.py`: aplicación interactiva en Streamlit.
- `Comparacion_Modelos_Imagenes_Hito2.ipynb`: entrenamiento y comparación de EfficientNet-B0 y YOLO26n-cls.
- `Aplicacion_Control_Calidad_Hito2.ipynb`: desarrollo y pruebas de la aplicación.
- `api/` y `Dockerfile`: servicio de clasificación utilizado desde n8n.
- `modelos/`: modelo EfficientNet-B0 entrenado.
- `resultados/`: métricas de prueba y selección del modelo.
- `ejemplos/`: fotografías para probar la aplicación.

En las 186 imágenes de prueba, EfficientNet-B0 obtuvo una exactitud balanceada de 96,43%, frente al 91,70% de YOLO26n-cls.

## Cómo ejecutar Streamlit

Con Python instalado, desde la carpeta del proyecto:

```bash
pip install -r requirements.txt
streamlit run app.py
```

La clasificación funciona con el modelo incluido en `modelos/`. Para utilizar las recomendaciones de Gemini, hay que configurar `GEMINI_API_KEY` en los secretos de Streamlit y, si corresponde, `GEMINI_MODEL`. No se deben subir claves al repositorio.

## Alcance del proyecto

Los registros de producción son datos históricos de referencia y las fotografías provienen de un dataset público de Kaggle. La relación entre las fotos y las líneas de producción es simulada, por lo que las clasificaciones no comprueban defectos reales en una planta. Las recomendaciones son un apoyo para la inspección y requieren revisión humana.

## Uso de inteligencia artificial

Utilizamos herramientas de IA como complemento durante el desarrollo, principalmente para facilitar tareas de programación, corregir errores y mejorar la presentación de la aplicación y los informes. No fueron el único medio de revisión: contrastamos los resultados con el código, las métricas y las pruebas realizadas. Gemini también forma parte del sistema para generar orientaciones y recomendaciones a partir de los resultados obtenidos.
