import cv2
import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

st.set_page_config(page_title="Detección de Objetos en Imágenes (YOLOv5)", page_icon="🔎", layout="wide")
st.title("🔎 Detección de Objetos en Imágenes (YOLOv5)")
st.markdown("Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara. "
            "Ajusta los parámetros en la barra lateral para personalizar la detección.")

# ---------- Intentar cargar YOLOv5 si existe ----------
yolo_ok = False
model = None
label_names = None

try:
    import yolov5  # si el paquete está instalado
    try:
        model = yolov5.load("yolov5s.pt", weights_only=False)
    except TypeError:
        model = yolov5.load("yolov5s.pt")  # fallback de carga
    # Ajustes por defecto (los sliders los sobreescriben)
    model.conf = 0.25
    model.iou = 0.45
    label_names = model.names
    yolo_ok = True
except Exception as e:
    st.info("Paquete 'yolov5' no encontrado. Activando **modo demo sin PyTorch** (detección de caras con OpenCV).")

# ---------- Sidebar ----------
st.sidebar.header("Parámetros")
conf = st.sidebar.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
iou  = st.sidebar.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
st.sidebar.caption(f"Confianza: {conf:.2f} | IoU: {iou:.2f}")

# Si hay modelo YOLO, aplicamos sliders sobre el modelo
if yolo_ok and model is not None:
    try:
        model.conf = conf
        model.iou = iou
    except:
        pass

st.divider()

# ---------- Cámara ----------
picture = st.camera_input("Capturar imagen")

if picture is None:
    st.caption("Toma una foto para iniciar la detección.")
    st.stop()

# Convertir a OpenCV
bytes_data = picture.getvalue()
img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

# ---------- Detección ----------
col1, col2 = st.columns(2)

if yolo_ok and model is not None:
    # ---- YOLO real ----
    with st.spinner("Detectando con YOLOv5..."):
        try:
            results = model(img)  # inferencia
            # YOLOv5 (repo Ultralytics v5) expone results.xyxy / results.pred, usamos .xyxy[0] si existe
            try:
                det = results.xyxy[0]  # [x1,y1,x2,y2,conf,cls]
            except AttributeError:
                det = results.pred[0]
            det = det.cpu().numpy() if hasattr(det, "cpu") else np.array(det)

            # Dibujar cajas
            drawn = img.copy()
            data = []
            for *xyxy, conf_score, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                c = int(cls)
                label = label_names[c] if label_names and c in range(len(label_names)) else f"id_{c}"
                cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(drawn, f"{label} {conf_score:.2f}", (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                data.append({"Categoría": label, "Confianza": round(float(conf_score), 3)})

            with col1:
                st.subheader("Imagen con detecciones")
                st.image(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col2:
                st.subheader("Objetos detectados")
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                    counts = df.groupby("Categoría").size().rename("Cantidad")
                    st.bar_chart(counts)
                else:
                    st.info("No se detectaron objetos con los parámetros actuales.")

        except Exception as e:
            st.error(f"❌ Error durante la detección con YOLO: {e}")

else:
    # ---- Fallback OpenCV Haar (sin PyTorch) ----
    with st.spinner("Detectando caras (modo demo sin PyTorch)..."):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        drawn = img.copy()
        data = []
        for (x, y, w, h) in faces:
            cv2.rectangle(drawn, (x, y), (x+w, y+h), (255, 100, 0), 2)
            cv2.putText(drawn, "face (demo)", (x, max(0, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            data.append({"Categoría": "face (demo)", "Confianza": "—"})

        with col1:
            st.subheader("Imagen con detecciones (demo)")
            st.image(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Resultados")
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                counts = df.groupby("Categoría").size().rename("Cantidad")
                st.bar_chart(counts)
            else:
                st.info("No se detectaron caras. Intenta otra toma (frontal, con luz).")

st.markdown("---")
st.caption("Si agregas `yolov5` + `torch` compatibles con tu Python, la app cambiará automáticamente al modo YOLOv5.")
