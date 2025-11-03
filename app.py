import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Detección de Objetos — Taylor’s Version", page_icon="🔍", layout="wide")

@st.cache_resource
def load_model():
    """
    Carga YOLOv5 desde torch.hub.
    - Si existe un archivo 'yolov5s.pt' en la raíz, lo usa como pesos 'custom'.
    - Si no, usa el modelo pre-entrenado 'yolov5s'.
    """
    # Fuerza CPU en Streamlit Cloud
    device = "cpu"
    try:
        if os.path.exists("yolov5s.pt"):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', trust_repo=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
        model.to(device)
        return model
    except Exception as e:
        st.error(f"❌ Error cargando modelo YOLOv5: {e}")
        return None

st.title("🔍 Detección de Objetos en Imágenes (YOLOv5)")
st.markdown("Captura una foto con tu cámara y detecta objetos. Ajusta la **confianza** en la barra lateral.")

model = load_model()
if model is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.subheader("Parámetros")
    conf_thres = st.slider("Confianza mínima", 0.00, 1.00, 0.25, 0.01)
    iou_thres  = st.slider("Umbral IoU (NMS)", 0.00, 1.00, 0.45, 0.01)
    st.caption(f"Conf: {conf_thres:.2f} | IoU: {iou_thres:.2f}")

# aplica parámetros (si existen estos attrs en la versión cargada)
if hasattr(model, 'conf'):
    model.conf = conf_thres
if hasattr(model, 'iou'):
    model.iou = iou_thres

# Cámara
picture = st.camera_input("Capturar imagen", key="camera")
if not picture:
    st.info("Toma una foto para empezar ✨")
    st.stop()

# Decodificar a OpenCV (BGR) y convertir a RGB para YOLOv5
bytes_data = picture.getvalue()
img_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("No se pudo decodificar la imagen.")
    st.stop()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Inferencia
with st.spinner("Detectando objetos..."):
    results = model(img_rgb)

# Render anotado
results.render()  # dibuja sobre results.ims
annotated = results.ims[0]  # numpy RGB

col1, col2 = st.columns(2)
with col1:
    st.subheader("Imagen con detecciones")
    st.image(annotated, use_container_width=True)

with col2:
    st.subheader("Objetos detectados")
    # Usamos el helper pandas() de YOLOv5 para extraer dataframe
    df_det = results.pandas().xyxy[0]  # columns: xmin,xmax,ymin,ymax,confidence,class,name
    if df_det.empty:
        st.info("No se detectaron objetos con los parámetros actuales. Prueba bajar la confianza.")
    else:
        # Resumen por clase
        resumen = (
            df_det.groupby("name")
            .agg(Cantidad=("name", "count"), Conf_Prom=("confidence", "mean"))
            .reset_index()
        )
        resumen["Conf_Prom"] = (resumen["Conf_Prom"] * 100).round(1).astype(str) + "%"
        st.dataframe(resumen, use_container_width=True)
        # Gráfico simple
        st.bar_chart(resumen.set_index("name")["Cantidad"])

st.markdown("---")
st.caption("App hecha con Streamlit + YOLOv5 (CPU).")
