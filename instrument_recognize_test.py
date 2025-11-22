import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.activations import softmax
import numpy as np
import os

# --- CONFIGURAÇÕES ---
modelo_path = 'modelo_instrument_recognize.h5'  # caminho do seu modelo
pasta_imagens = 'images_tests'                  # pasta com imagens de teste
altura, largura = 256, 256  # tamanho correto usado no treino

# --- CARREGAR MODELO COM CORREÇÃO DO SOFTMAX_V2 ---
custom_objects = {'softmax_v2': softmax}
model = load_model(modelo_path, custom_objects=custom_objects)
print("Modelo carregado com sucesso!")

# --- FUNÇÃO DE PREPARO DE IMAGEM ---
def preparar_imagem(img_path):
    img = image.load_img(img_path, target_size=(altura, largura))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # adiciona dimensão do batch
    img_array = img_array / 255.0                  # normalização (se usada no treino)
    return img_array

# Obtém as classes do dataset usado no treino
classes = ['analog', 'digital']  # ou use treino.class_names se ainda disponível

for nome_arquivo in os.listdir(pasta_imagens):
    caminho = os.path.join(pasta_imagens, nome_arquivo)
    if caminho.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_preparada = preparar_imagem(caminho)
        predicao = model.predict(img_preparada)
        indice = np.argmax(predicao, axis=1)[0]
        classe_prevista = classes[indice]
        print(f"Imagem: {nome_arquivo} -> Classe prevista: {classe_prevista}")
