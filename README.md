# Processo Seletivo - Intensivo Maker - IA

## Relatório

Nome Completo: Raíssa de Brito Duarte

# Arquitetura do Modelo

Implementei uma CNN com três blocos convolucionais para classificar os dígitos do MNIST.

A entrada são imagens de 28x28 pixels em escala de cinza, normalizadas entre 0 e 1. O modelo segue essa estrutura:

- Conv2D (32 filtros, 3x3) + MaxPooling: extrai as primeiras características das imagens
- Conv2D (64 filtros, 3x3) + MaxPooling: aprofunda a extração de padrões
- Conv2D (64 filtros, 3x3): mais uma camada convolucional antes de achatar
- Flatten + Dense (64) + Dropout (0.3): classifica com base nas features extraídas
- Dense (10, softmax): saída com a probabilidade de cada dígito (0 a 9)

Usei `adam` como otimizador e `sparse_categorical_crossentropy` como função de perda. O treinamento foi feito com 5 épocas e batch de 128, respeitando o limite de tempo do CI.


# Bibliotecas Utilizadas

- TensorFlow/Keras: utilizado para criação, treinamento e salvamento do modelo; 
- os: utilizado para verificar tamanho dos arquivos gerados.

As versões estão no `requirements.txt`.


# Técnica de Otimização

Usei **Dynamic Range Quantization** no `optimize_model.py`.

Essa técnica converte os pesos do modelo de `float32` pra `int8` na hora da conversão, sem precisar de dados de calibração. O resultado é um modelo bem menor, adequado pra rodar em dispositivos com pouca memória — como microcontroladores e sistemas embarcados.

Apliquei com `converter.optimizations = [tf.lite.Optimize.DEFAULT]`, que é a forma mais simples e direta de fazer isso no TFLite.


# Resultados

- Acurácia no conjunto de teste: ~ 98%
- Arquivo gerado: `model.h5`
- Arquivo otimizado: `model.tflite`
- Redução de tamanho: aproximadamente 70%


# Comentários

Esse desafio foi bem interessante porque mostrou o fluxo completo de um modelo de IA voltado pra Edge: treinar, salvar, converter e otimizar. Eu já tinha visto regressão e séries temporais no curso, mas trabalhar com CNN no MNIST foi a parte mais prática até agora.

O maior cuidado foi manter a arquitetura simples o suficiente pra rodar dentro do tempo do pipeline de CI, sem perder acurácia. O Dropout ajudou a evitar overfitting sem complicar o modelo.