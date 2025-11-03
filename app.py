import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st

# Configuración de página Streamlit
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# Cargador del modelo YOLOv5 con fallback
@st.cache_resource
def load_yolov5_model(model_path="yolov5s.pt"):
    """
    1) Intenta usar el paquete 'yolov5' si existe (y soporta weights_only).
    2) Si falla, usa torch.hub con 'ultralytics/yolov5' (no requiere instalar ultralytics/yolov5 de PyPI).
    """
    # 1) Intento con paquete 'yolov5'
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            # Algunas versiones no aceptan weights_only
            model = yolov5.load(model_path)
            return model
        except Exception as e:
            st.warning(f"Fallo al cargar con paquete 'yolov5' ({e}). Probando torch.hub…")
    except Exception:
        st.info("Paquete 'yolov5' no encontrado. Usando torch.hub como respaldo…")

    # 2) Fallback con torch.hub (pretrained CPU)
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
        return model
    except Exception as e2:
        st.error(f"❌ No se pudo cargar el modelo vía torch.hub: {e2}")
        return None


# Título y descripción
st.title("🔍 Detección de Objetos en Imágenes (YOLOv5)")
st.markdown("""
Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara.
Ajusta los parámetros en la barra lateral para personalizar la detección.
""")

# Cargar modelo
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if not model:
    st.error("No se pudo cargar el modelo. Verifica dependencias e inténtalo nuevamente.")
    st.stop()

# Sidebar de parámetros
st.sidebar.title("Parámetros")
with st.sidebar:
    st.subheader("Configuración de detección")
    conf = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
    st.caption(f"Confianza: {conf:.2f} | IoU: {iou:.2f}")

    # Aplica parámetros si el objeto modelo los soporta
    if hasattr(model, "conf"):
        model.conf = conf
    if hasattr(model, "iou"):
        model.iou = iou

    st.subheader("Opciones avanzadas")
    try:
        model.agnostic = st.checkbox("NMS class-agnostic", False)
        model.multi_label = st.checkbox("Múltiples etiquetas por caja", False)
        model.max_det = st.number_input("Detecciones máximas", 10, 2000, 1000, 10)
    except Exception:
        st.warning("Algunas opciones avanzadas no están disponibles con esta configuración.")

# Captura de cámara
picture = st.camera_input("Capturar imagen", key="camera")

if picture:
    # Decodificar a OpenCV (BGR)
    bytes_data = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    if cv2_img is None:
        st.error("No se pudo decodificar la imagen.")
        st.stop()

    # Inferencia
    with st.spinner("Detectando objetos..."):
        try:
            results = model(cv2_img)  # YOLOv5 acepta BGR/RGB; internamente convierte
        except Exception as e:
            st.error(f"Error durante la detección: {e}")
            st.stop()

    # Render anotado
    try:
        results.render()            # Dibuja sobre results.ims
        annotated = results.ims[0]  # Imagen anotada (numpy RGB)
    except Exception:
        annotated = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # Fallback: sin anotaciones

    # Layout de resultados
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen con detecciones")
        st.image(annotated, use_container_width=True)

    with col2:
        st.subheader("Objetos detectados")
        try:
            # Extraer predicciones (tensor) si está disponible
            predictions = results.pred[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            categories = predictions[:, 5]

            # Nombres de clases
            label_names = getattr(model, "names", None)
            if label_names is None and hasattr(results, "names"):
                label_names = results.names

            # Conteo por clase
            category_count = {}
            for cat in categories:
                idx = int(cat.item()) if hasattr(cat, "item") else int(cat)
                category_count[idx] = category_count.get(idx, 0) + 1

            data = []
            for idx, count in category_count.items():
                label = label_names[idx] if label_names and idx in label_names else f"cls_{idx}"
                mask = (categories == idx)
                conf_avg = scores[mask].mean().item() if mask.any() else 0.0
                data.append({"Categoría": label, "Cantidad": count, "Confianza promedio": f"{conf_avg:.2f}"})

            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("Categoría")["Cantidad"])
            else:
                st.info("No se detectaron objetos con los parámetros actuales. Prueba a reducir la confianza.")
        except Exception as e:
            # Alternativa: usar el helper pandas() de YOLOv5 si existe
            try:
                df_det = results.pandas().xyxy[0]
                if df_det.empty:
                    st.info("No se detectaron objetos con los parámetros actuales.")
                else:
                    resumen = (
                        df_det.groupby("name")
                        .agg(Cantidad=("name", "count"), Conf_Prom=("confidence", "mean"))
                        .reset_index()
                    )
                    resumen["Conf_Prom"] = (resumen["Conf_Prom"] * 100).round(1).astype(str) + "%"
                    st.dataframe(resumen, use_container_width=True)
                    st.bar_chart(resumen.set_index("name")["Cantidad"])
            except Exception as e2:
                st.error(f"Error al procesar resultados: {e} | {e2}")

# Pie de página
st.markdown("---")
st.caption("**Acerca de la aplicación**: Detección de objetos en tiempo real con YOLOv5. Desarrollada con Streamlit y PyTorch.")
