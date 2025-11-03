import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="Detección de Objetos — Taylor’s Version",
    page_icon="🔍",
    layout="wide"
)

# Función para cargar el modelo YOLOv5 de manera compatible con versiones anteriores de PyTorch
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        # Importar yolov5
        import yolov5
        
        # Para versiones de PyTorch anteriores a 2.0, cargar directamente con weights_only=False
        # o usar el parámetro map_location para asegurar compatibilidad
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception as e:
                st.warning("Intentando método alternativo de carga (torch.hub)…")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # Fallback vía torch.hub (no requiere paquete yolov5/ultralytics instalado)
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
                return model
    
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1) Verifica dependencias en requirements.
        2) Asegúrate de tener el archivo de pesos si vas a usar `model_path`.
        3) Si falla la importación del módulo `yolov5`, el fallback usa `torch.hub`.
        """)
        return None

# Título y descripción de la aplicación
st.title("🔍 Detector de Objetos (Taylor’s Version)")
st.markdown("""
Sube una foto con tu **cámara** y YOLOv5 identificará lo que aparece.  
Ajusta los **umbrales** en el panel lateral para afinar resultados.
""")

# Cargar el modelo
with st.spinner("Cargando modelo YOLOv5…"):
    model = load_yolov5_model()

# Si el modelo se cargó correctamente, configuramos los parámetros
if model:
    # Sidebar para los parámetros de configuración
    st.sidebar.title("Parámetros")
    
    # Ajustar parámetros del modelo
    with st.sidebar:
        st.subheader('Configuración de detección')
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")
        
        # Opciones adicionales
        st.subheader('Opciones avanzadas')
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("Algunas opciones avanzadas no están disponibles con esta configuración")
    
    # Contenedor principal para la cámara y resultados
    main_container = st.container()
    
    with main_container:
        # Capturar foto con la cámara
        picture = st.camera_input("Capturar imagen", key="camera")
        
        if picture:
            # Procesar la imagen capturada
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Realizar la detección
            with st.spinner("Detectando objetos…"):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()
            
            # Parsear resultados
            try:
                predictions = results.pred[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]
                
                # Mostrar resultados
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Imagen con detecciones")
                    results.render()  # dibuja sobre results.ims
                    st.image(cv2_img, channels='BGR', use_container_width=True)
                
                with col2:
                    st.subheader("Objetos detectados")
                    
                    # Nombres de etiquetas
                    label_names = model.names
                    
                    # Conteo por categoría
                    category_count = {}
                    for category in categories:
                        category_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                        category_count[category_idx] = category_count.get(category_idx, 0) + 1
                    
                    # Tabla de resumen
                    data = []
                    for category, count in category_count.items():
                        label = label_names[category]
                        confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                        data.append({
                            "Categoría": label,
                            "Cantidad": count,
                            "Confianza promedio": f"{confidence:.2f}"
                        })
                    
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index('Categoría')['Cantidad'])
                    else:
                        st.info("No se detectaron objetos con los parámetros actuales.")
                        st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")
            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
                st.stop()
else:
    st.error("No se pudo cargar el modelo. Verifica dependencias e inténtalo nuevamente.")
    st.stop()

@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception:
                st.warning("Intentando método alternativo de carga (torch.hub)…")
                # Fallback: NO requiere tener instalado 'yolov5' o 'ultralytics'
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
                return model
    except Exception:
        # Si el import yolov5 falla, haz fallback directo
        st.warning("No se encontró el paquete 'yolov5'. Usando torch.hub como respaldo…")
        try:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
            return model
        except Exception as e2:
            st.error(f"❌ Error al cargar el modelo vía torch.hub: {e2}")
            return None


# Pie
st.markdown("---")
st.caption("App con Streamlit + YOLOv5. Narrativa: Taylor’s Version 💫")
