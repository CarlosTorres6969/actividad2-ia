import tensorflow as tf
from pathlib import Path

model_path = Path("modelo_reciclaje_mobilenet/waste_mobilenet.h5")
output_path = Path("modelo_reciclaje_mobilenet/waste_mobilenet.tflite")

print("Cargando modelo...")
model = tf.keras.models.load_model(str(model_path), compile=False)

print("Convirtiendo a TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

output_path.write_bytes(tflite_model)
print(f"Modelo guardado en {output_path} ({len(tflite_model) / 1024 / 1024:.2f} MB)")
