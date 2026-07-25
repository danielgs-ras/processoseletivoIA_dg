import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

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