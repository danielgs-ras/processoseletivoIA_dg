## 📝 Relatório do Candidato

👤 **Nome Completo:** Daniel Gustavo Rodrigues Andrade Souza

### 1️⃣ Resumo da Arquitetura do Modelo

Projeto escolhido: **Classificação MNIST**

Foi implementada uma Rede Neural Convolucional (CNN) para classificação de dígitos manuscritos do conjunto de dados MNIST.

A arquitetura é composta por três blocos convolucionais, sendo que cada bloco utiliza uma camada `Conv2D`, seguida por `BatchNormalization` e `MaxPooling2D`. Após a extração das características, os mapas gerados são convertidos em um vetor utilizando `Flatten`. Em seguida, é aplicada uma camada `Dropout` com taxa de 0,5 para reduzir o risco de overfitting antes da camada de saída.

A camada final possui 10 neurônios com ativação `softmax`, responsáveis pela classificação dos dígitos de 0 a 9.

Para o treinamento foi utilizado `validation_split=0.1` para separar o conjunto de validação e o callback `EarlyStopping`, monitorando `val_loss` com restauração automática dos melhores pesos obtidos durante o treinamento.

### 2️⃣ Bibliotecas Utilizadas

Principais bibliotecas utilizadas:

- TensorFlow == 2.21.0
- Keras == 3.12.3
- NumPy == 2.2.6
- os (módulo padrão da biblioteca do Python)

### 3️⃣ Técnica de Otimização do Modelo

Após o treinamento, o modelo salvo em `model.h5` foi convertido para TensorFlow Lite (`model.tflite`).

Durante essa conversão foi aplicada a técnica **Dynamic Range Quantization**, utilizando:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

Essa técnica reduz o tamanho do modelo por meio da quantização automática dos pesos, tornando-o mais adequado para execução em dispositivos de Edge AI, mantendo desempenho adequado para a tarefa proposta.

### 4️⃣ Resultados Obtidos

- **Acurácia final de validação:** 99,00%
- **Acurácia final no conjunto de teste:** 98,89%
- **Tamanho do arquivo `model.h5`:** 1,3 MB
- **Tamanho do arquivo `model.tflite`:** 114 KB

A conversão para TensorFlow Lite reduziu significativamente o tamanho do modelo, mantendo elevada acurácia durante os testes de inferência.

### 5️⃣ Comentários Adicionais (Opcional)

A arquitetura foi desenvolvida buscando equilibrar desempenho e simplicidade, utilizando três blocos convolucionais para extração de características e técnicas de regularização, como `BatchNormalization` e `Dropout`, para tornar o treinamento mais estável.

A utilização da técnica de Dynamic Range Quantization permitiu reduzir significativamente o tamanho do modelo otimizado, tornando-o mais adequado para aplicações em dispositivos embarcados sem comprometer o desempenho observado durante os testes realizados.

### 6️⃣ Exemplo de Inferência

```text
Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```

Nas amostras avaliadas, todas as previsões corresponderam às classes reais, indicando que o modelo convertido para TensorFlow Lite manteve o comportamento esperado após a etapa de otimização.