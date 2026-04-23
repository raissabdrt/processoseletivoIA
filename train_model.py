import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# carregando o dataset MNIST direto do keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# normalizando os valores dos pixels pra ficar entre 0 e 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# adicionando a dimensão do canal (grayscale = 1 canal)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# construindo o modelo CNN
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # primeiro bloco convolucional
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # segundo bloco convolucional
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # terceiro bloco convolucional
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),

    # achata pra entrar na camada densa
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),

    # saída com 10 classes (dígitos de 0 a 9)
    layers.Dense(10, activation="softmax")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# treinando por 5 épocas conforme o limite do desafio
print("\nIniciando treinamento...\n")
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# avaliando no conjunto de teste com duas métricas: loss e accuracy
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f"\nResultados no conjunto de teste:")
print(f"  Acurácia : {test_acc * 100:.2f}%")
print(f"  Loss     : {test_loss:.4f}")
print(f"\n  A acurácia indica o percentual de dígitos classificados corretamente.")
print(f"  O loss indica o erro médio do modelo — quanto menor, melhor.")

# salvando o modelo
model.save("model.h5")
print("\nModelo salvo em model.h5")