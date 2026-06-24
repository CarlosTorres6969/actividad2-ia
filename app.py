import streamlit as st
import numpy as np
from PIL import Image
import json
from pathlib import Path

try:
    import tflite_runtime.interpreter as tflite
    USE_TFLITE = True
except ImportError:
    import tensorflow as tf
    USE_TFLITE = False

st.set_page_config(
    page_title="Clasificador de Residuos",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Clasificador de Residuos con IA")
st.write("Sube una imagen de un residuo y la IA te dirá de qué material es.")
st.caption("Desarrollado por **José Carlos Torres**")

@st.cache_resource
def load_model():
    tflite_path = Path("modelo_reciclaje_mobilenet/waste_mobilenet.tflite")
    h5_path = Path("modelo_reciclaje_mobilenet/waste_mobilenet.h5")

    if USE_TFLITE and tflite_path.exists():
        interpreter = tflite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        return ("tflite", interpreter)
    elif not USE_TFLITE and h5_path.exists():
        model = tf.keras.models.load_model(str(h5_path), compile=False)
        return ("keras", model)
    else:
        return (None, None)

@st.cache_resource
def load_class_names():
    json_path = Path("modelo_reciclaje_mobilenet/class_names.json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Basura",
}

IMG_SIZE = (224, 224)

model_type, model = load_model()
class_names = load_class_names()

uploaded_file = st.file_uploader(
    "Elige una imagen...",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    img_resized = image.resize(IMG_SIZE)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    with st.spinner("Analizando..."):
        if model_type == "tflite":
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            model.set_tensor(input_details[0]['index'], arr)
            model.invoke()
            preds = model.get_tensor(output_details[0]['index'])[0]
        elif model_type == "keras":
            preds = model.predict(arr, verbose=0)[0]
        else:
            st.error("No se pudo cargar el modelo.")
            st.stop()

    st.subheader("Resultados:")

    top3_idx = np.argsort(preds)[-3:][::-1]

    for idx in top3_idx:
        raw = class_names[idx]
        nombre_es = LABELS_ES.get(raw, raw)
        prob = preds[idx] * 100
        st.progress(prob / 100)
        st.write(f"**{nombre_es}**: {prob:.2f}%")

    top_class = class_names[np.argmax(preds)]
    st.success(f"**Predicción principal: {LABELS_ES.get(top_class, top_class)}**")

else:
    st.info("Sube una imagen para comenzar la clasificación.")

st.sidebar.header("Acerca de")
st.sidebar.info(
    "Este clasificador utiliza un modelo MobileNetV2 "
    "entrenado con imágenes de residuos reciclables."
)
st.sidebar.header("Clases:")
for clase in class_names if class_names else []:
    st.sidebar.write(f"- {LABELS_ES.get(clase, clase)}")
