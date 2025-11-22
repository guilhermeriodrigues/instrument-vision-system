import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 0) Verificação da GPU — agora apenas informativo
# -------------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPUs detectadas:", gpus)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Aviso ao configurar GPU:", e)
else:
    print("Nenhuma GPU detectada. O treinamento será feito na CPU.")

# -------------------------------------------------------------
# 1) Caminhos e configurações
# -------------------------------------------------------------
data_dir = Path("C:/Users/guilh/Documents/instrument-vision-system/images_augmented")

batch_size = 32
altura = 256
largura = 256
seed_global = 150

# -------------------------------------------------------------
# 2) Carregar dataset
# -------------------------------------------------------------
treino = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=seed_global,
    image_size=(altura, largura),
    batch_size=batch_size
)

validacao = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=seed_global,
    image_size=(altura, largura),
    batch_size=batch_size
)

print("Classes detectadas:", treino.class_names)

# -------------------------------------------------------------
# 3) Normalização + Augmentation
# -------------------------------------------------------------
normalizar = tf.keras.layers.Rescaling(1./255)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),   # 5% ≈ 9°
    tf.keras.layers.RandomZoom(0.1),
])

# treinar: normaliza + augmenta
treino = treino.map(
    lambda x, y: (data_augmentation(normalizar(x), training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# validação: apenas normaliza
validacao = validacao.map(
    lambda x, y: (normalizar(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Cache + prefetch → aceleração no CPU
treino = treino.cache().prefetch(tf.data.AUTOTUNE)
validacao = validacao.cache().prefetch(tf.data.AUTOTUNE)

print("Dataset pronto para treino!")

# -------------------------------------------------------------
# 4) CNN otimizada
# -------------------------------------------------------------
modelo = tf.keras.models.Sequential([
 tf.keras.layers.Input(shape=(256,256,3)),
 tf.keras.layers.Rescaling(1./255),
 tf.keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu),
 tf.keras.layers.MaxPooling2D(2,2),
 tf.keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu),
 tf.keras.layers.MaxPooling2D(2,2),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(128, activation=tf.nn.relu),
 tf.keras.layers.Dense(4, activation=tf.nn.softmax)
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

modelo.summary()

# -------------------------------------------------------------
# 5) Treinamento
# -------------------------------------------------------------
epocas = 20

class myCallback (tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>=0.93):
            print('\n Alcançamos 93% de acurácia. Parando o treinamento')
            self.model.stop_training = True

callbacks = myCallback()

history = modelo.fit(
    treino,
    validation_data=validacao,
    epochs=epocas,
    callbacks=[callbacks]
)

modelo.save('modelo_instrument_recognize.h5')

# -------------------------------------------------------------
# 6) Gráfico de resultados
# -------------------------------------------------------------
def plota_resultados(history, epocas):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    intervalo_epocas = range(epocas)

    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.plot(intervalo_epocas, acc, 'r', label='Treino')
    plt.plot(intervalo_epocas, val_acc, 'b', label='Validação')
    plt.title("Acurácia")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(intervalo_epocas, loss, 'r', label='Treino')
    plt.plot(intervalo_epocas, val_loss, 'b', label='Validação')
    plt.title("Perda")
    plt.legend()

    plt.show()

plota_resultados(history, epocas)