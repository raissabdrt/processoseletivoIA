import tensorflow as tf
import os

# carregando o modelo treinado
print("Carregando o modelo...")
model = tf.keras.models.load_model("model.h5")
print("Modelo carregado!")

# criando o conversor pra TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# aplicando dynamic range quantization
# isso reduz o tamanho do modelo convertendo os pesos de float32 pra int8
# boa opção pra rodar em dispositivos embarcados com pouca memória
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# convertendo
print("\nConvertendo para TFLite com quantização...")
tflite_model = converter.convert()

# salvando o arquivo .tflite
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# comparando os tamanhos
tamanho_original = os.path.getsize("model.h5") / 1024
tamanho_otimizado = os.path.getsize("model.tflite") / 1024

print(f"\nModelo original (.h5):     {tamanho_original:.1f} KB")
print(f"Modelo otimizado (.tflite): {tamanho_otimizado:.1f} KB")
print(f"Redução de tamanho: {100 * (1 - tamanho_otimizado / tamanho_original):.1f}%")
print("\nModelo salvo em model.tflite")