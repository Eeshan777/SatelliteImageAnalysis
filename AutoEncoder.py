import tensorflow as tf
from tensorflow.keras import layers, models
import os

IMG_SIZE = (512, 512)
BATCH_SIZE = 8
STEPS = 800
EPOCHS = 30
DATA_PATH = "./dataset_extracted"

# REPEAT is critical for custom steps_per_epoch
dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None
).map(lambda x: (x/255.0, x/255.0)).repeat().prefetch(tf.data.AUTOTUNE)

# ACTUAL AUTOENCODER (Encoder-Decoder)
def build_autoencoder():
    inputs = layers.Input(shape=(512, 512, 3))
    
    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(inputs)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
    
    # Bottleneck
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    # Decoder
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_autoencoder()
print("Starting Autoencoder Training...")
model.fit(dataset, epochs=EPOCHS, steps_per_epoch=STEPS)
model.save("models/autoencoder_model.keras")