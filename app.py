import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Residuos",
    page_icon="♻️",
    layout="centered"
)

# Título y descripción
st.title("♻️ Clasificador de Residuos con IA")
st.write("Sube una imagen de un residuo y la IA te dirá de qué material es.")

# Cargar modelo y clases
@st.cache_resource
def load_model():
    model_path = Path("modelo_reciclaje_mobilenet/waste_mobilenet.h5")
    if model_path.exists():
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    else:
        st.error("No se encontró el modelo. Ejecuta primero el notebook de entrenamiento.")
        return None

@st.cache_resource
def load_class_names():
    json_path = Path("modelo_reciclaje_mobilenet/class_names.json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        st.error("No se encontró el archivo de clases.")
        return None

# Diccionario de traducción
LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Basura",
}

IMG_SIZE = (224, 224)

# Cargar recursos
model = load_model()
class_names = load_class_names()

if model and class_names:
    # Interfaz de usuario
    uploaded_file = st.file_uploader(
        "Elige una imagen...",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        # Mostrar imagen
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen subida", use_column_width=True)

        # Preprocesar imagen
        img_resized = image.resize(IMG_SIZE)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Hacer predicción
        with st.spinner("Analizando..."):
            preds = model.predict(arr, verbose=0)[0]

        # Mostrar resultados
        st.subheader("Resultados:")

        # Top 3 predicciones
        top3_idx = np.argsort(preds)[-3:][::-1]

        for idx in top3_idx:
            raw = class_names[idx]
            nombre_es = LABELS_ES.get(raw, raw)
            prob = preds[idx] * 100

            # Barra de progreso
            st.progress(prob / 100)
            st.write(f"**{nombre_es}**: {prob:.2f}%")

        # Predicción principal
        top_class = class_names[np.argmax(preds)]
        st.success(f"**Predicción principal: {LABELS_ES.get(top_class, top_class)}**")

    else:
        st.info("Sube una imagen para comenzar la clasificación.")

# Información lateral
st.sidebar.header(" Acerca de")
st.sidebar.info(
    "Este clasificador utiliza un modelo MobileNetV2 "
    "entrenado con imágenes de residuos reciclables."
)
st.sidebar.header(" Clases:")
for clase in class_names if class_names else []:
    st.sidebar.write(f"- {LABELS_ES.get(clase, clase)}")
