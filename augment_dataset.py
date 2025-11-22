import os
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# -------------------------------------------
# CONFIGURAÇÕES
# -------------------------------------------
INPUT_DIR = Path("images")
OUTPUT_DIR = Path("images_augmented")
IMG_SIZE = (224, 224)
AUG_PER_IMAGE = 5   # Quantidade de imagens aumentadas por imagem original

# Cria diretório de saída mantendo a estrutura
for class_dir in ["Analog_Instruments", "Digital_Instruments"]:
    out_path = OUTPUT_DIR / class_dir
    out_path.mkdir(parents=True, exist_ok=True)


# -------------------------------------------
# AUGMENTATION REALISTA PARA INSTRUMENTOS
# -------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomBrightness(factor=0.20),
    tf.keras.layers.RandomContrast(factor=0.20),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(factor=5/360, fill_mode="nearest"),
    tf.keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10),
])


def augment_image(img):
    """
    img: tf.Tensor uint8 [0,255], shape (H,W,3)
    retorna tf.Tensor uint8 [0,255], com augmentação segura
    """
    img = tf.cast(img, tf.float32) / 255.0  # normaliza 0-1

    # Adiciona batch dim para usar layers do Keras
    img = tf.expand_dims(img, 0)

    # Transformações geométricas pequenas
    geom_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.02, fill_mode="reflect"),  # ±2%
    ])
    img = geom_aug(img, training=True)

    # Transformações de cor suaves
    img = tf.image.random_brightness(img, max_delta=0.05)  # ±5%
    img = tf.image.random_contrast(img, lower=0.95, upper=1.05)

    # Remove batch dim
    img = tf.squeeze(img, 0)

    # Garante que o tensor está no intervalo 0-1 antes de converter
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.cast(img * 255.0, tf.uint8)

    return img



# -------------------------------------------
# CARREGAR IMAGEM
# -------------------------------------------
def load_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    return np.array(img)


# -------------------------------------------
# PROCESSAR TODO O DATASET
# -------------------------------------------
def process_dataset():

    for class_name in ["Analog_Instruments", "Digital_Instruments"]:
        class_input_dir = INPUT_DIR / class_name
        class_output_dir = OUTPUT_DIR / class_name

        print(f"\n📂 Processando classe: {class_name}")

        for filename in os.listdir(class_input_dir):

            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            input_path = class_input_dir / filename
            print(f" - Carregando: {filename}")

            img_np = load_image(input_path)
            img_tensor = tf.convert_to_tensor(img_np)

            # Gera várias imagens por arquivo
            for i in range(AUG_PER_IMAGE):

                aug_img = augment_image(img_tensor)
                aug_np = aug_img.numpy()

                output_filename = f"{filename.split('.')[0]}_aug{i}.png"
                output_path = class_output_dir / output_filename

                Image.fromarray(aug_np).save(output_path)

                print(f"   -> Gerado {output_filename}")


if __name__ == "__main__":
    process_dataset()

