import tensorflow as tf
import os

# Conversão do modelo treinado (model.h5) para TensorFlow Lite,
# aplicando Dynamic Range Quantization para reduzir o tamanho do arquivo.


script_dir = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(
    os.path.join(script_dir, "model.h5")
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Dynamic Range Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(os.path.join(script_dir, "model.tflite"), "wb") as f:
    f.write(tflite_model)

print("Modelo convertido e salvo como model.tflite")