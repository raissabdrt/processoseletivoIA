# Processo Seletivo - Intensivo Maker - IA

# Relatório

Nome Completo: Raíssa de Brito Duarte

# Arquitetura do Modelo

Implementei uma CNN com três blocos convolucionais para classificar os dígitos do MNIST.

A entrada são imagens de 28x28 pixels em escala de cinza, normalizadas entre 0 e 1. O modelo segue essa estrutura:

- Conv2D (32 filtros, 3x3) + MaxPooling: extrai as primeiras características das imagens
- Conv2D (64 filtros, 3x3) + MaxPooling: aprofunda a extração de padrões
- Conv2D (64 filtros, 3x3): mais uma camada convolucional antes de achatar
- Flatten + Dense (64) + Dropout (0.3): classifica com base nas features extraídas
- Dense (10, softmax): saída com a probabilidade de cada dígito (0 a 9)

Escolhi essa arquitetura porque é simples o suficiente pra rodar em CPU dentro do tempo do CI, mas ainda captura bem os padrões visuais do MNIST. O Dropout de 0.3 ajuda a evitar overfitting sem adicionar complexidade desnecessária.

Usei `adam` como otimizador e `sparse_categorical_crossentropy` como função de perda. O treinamento foi feito com 5 épocas e batch de 128.


# Bibliotecas Utilizadas

- TensorFlow/Keras: criação, treinamento, salvamento e conversão do modelo
- os: verificar e comparar o tamanho dos arquivos gerados

As versões estão no `requirements.txt`.


# Técnica de Otimização

Apliquei duas técnicas de quantização no `optimize_model.py`.

- Dynamic Range Quantization (técnica principal):

Converte os pesos de `float32` para `int8` na hora da conversão, sem precisar de dados de calibração. É a opção mais simples e gera a maior redução de tamanho (aproximadamente 70%). O arquivo gerado é o `model.tflite`.

- Float16 Quantization (técnica comparativa):

Converte os pesos para `float16` em vez de `int8`. Mantém mais precisão numérica que o DRQ, com redução de tamanho similar (aproximadamente 50%). Boa opção quando o dispositivo suporta operações em float16 e a precisão é mais crítica.

O DRQ foi escolhido como arquivo principal por ser mais compatível com microcontroladores com recursos muito limitados. O Float16 equilibra melhor tamanho e qualidade quando há mais memória disponível.


# Resultados

- Acurácia no conjunto de teste: ~98%
- Loss no conjunto de teste: ~0.06
- Arquivo gerado: `model.h5`
- Modelo DRQ: `model.tflite` (aproximadamente 70% menor que o original)
- Modelo Float16: `model_fp16.tflite` (aproximadamente 50% menor que o original)

A acurácia de ~98% indica que o modelo classifica corretamente quase todos os dígitos. O loss baixo confirma que o erro médio das previsões é pequeno, o que é esperado para o MNIST com uma CNN bem configurada.


# Comentários

Esse desafio foi bem interessante porque mostrou o fluxo completo de um modelo de IA voltado pra Edge: treinar, salvar, converter e otimizar. Eu já tinha visto regressão e séries temporais no curso, mas trabalhar com CNN no MNIST foi a parte mais prática até agora.

O maior cuidado foi manter a arquitetura simples o suficiente pra rodar dentro do tempo do pipeline de CI. Explorar as duas técnicas de quantização foi o que mais me ensinou nessa etapa, deu pra entender na prática que não existe uma única solução: depende do dispositivo, da memória disponível e de quanto de precisão será necessário ou não. 