import numpy as np
import tensorflow as tf

# Executa inferência com o modelo otimizado (model.tflite) em 5 amostras
# do conjunto de teste, comparando classe predita com classe real.


N_SAMPLES = 5


def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    interpreter = tf.lite.Interpreter(model_path=os.path.join(script_dir, "model.tflite"))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"Rodando inferencia em {N_SAMPLES} amostras usando model.tflite:\n")
    for i in range(N_SAMPLES):
        sample = np.expand_dims(x_test[i], axis=0).astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0]
        predicted_class = int(np.argmax(pred))
        print(f"Amostra {i + 1}: predito={predicted_class} | real={int(y_test[i])}")


if __name__ == "__main__":
    main()
