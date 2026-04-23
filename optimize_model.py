import tensorflow as tf
import os

# carregando o modelo treinado
print("Carregando o modelo...")
model = tf.keras.models.load_model("model.h5")
print("Modelo carregado!")

# Técnica 1: Dynamic Range Quantization
# converte os pesos de float32 pra int8 sem precisar de dados de calibração
# é a técnica mais simples e já gera uma boa redução de tamanho
print("\n[1] Aplicando Dynamic Range Quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_drq = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_drq)

tamanho_original = os.path.getsize("model.h5") / 1024
tamanho_drq = os.path.getsize("model.tflite") / 1024

print(f"  Modelo original (.h5)          : {tamanho_original:.1f} KB")
print(f"  Modelo DRQ (.tflite)           : {tamanho_drq:.1f} KB")
print(f"  Redução                        : {100 * (1 - tamanho_drq / tamanho_original):.1f}%")

# Técnica 2: Float16 Quantization 
# converte os pesos pra float16 em vez de int8
# mantém mais precisão que o DRQ, com redução de tamanho similar (~50%)
# boa opção quando se quer equilibrar tamanho e qualidade do modelo
print("\n[2] Aplicando Float16 Quantization...")
converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
converter2.optimizations = [tf.lite.Optimize.DEFAULT]
converter2.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter2.convert()

with open("model_fp16.tflite", "wb") as f:
    f.write(tflite_fp16)

tamanho_fp16 = os.path.getsize("model_fp16.tflite") / 1024

print(f"  Modelo Float16 (model_fp16.tflite): {tamanho_fp16:.1f} KB")
print(f"  Redução                           : {100 * (1 - tamanho_fp16 / tamanho_original):.1f}%")

# Comparativo final
print("\n Comparativo de tamanho ")
print(f"  Original (.h5)      : {tamanho_original:.1f} KB")
print(f"  DRQ (.tflite)       : {tamanho_drq:.1f} KB  ← melhor compressão")
print(f"  Float16 (.tflite)   : {tamanho_fp16:.1f} KB  ← melhor equilíbrio precisão/tamanho")
print("\nArquivo principal salvo em: model.tflite (Dynamic Range Quantization)")