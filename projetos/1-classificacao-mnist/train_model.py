import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Treinamento de uma CNN para classificação de dígitos do MNIST,
# com split de validação, EarlyStopping e salvamento em model.h5.


# Carregar dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# Normalizar imagens para [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# Ajustar formato para CNN: (28,28,1)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


# Construção da CNN
model = keras.Sequential([
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        input_shape=(28, 28, 1)
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dropout(0.5),

    layers.Dense(10, activation="softmax")
])


# Compilação
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# Treinamento
history = model.fit(
    x_train,
    y_train,
    epochs=15,
    validation_split=0.1,
    callbacks=[early_stop]
)


# Resultado da validação
val_accuracy = history.history["val_accuracy"][-1]

print(f"Acurácia final de validação: {val_accuracy:.4f}")


# Avaliação no conjunto de teste
loss, accuracy = model.evaluate(x_test, y_test)

print(f"Acurácia final no teste: {accuracy:.4f}")


# Salvar modelo treinado
script_dir = os.path.dirname(os.path.abspath(__file__))

model.save(os.path.join(script_dir, "model.h5"))

print("Modelo salvo como model.h5")