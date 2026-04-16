import os
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing import image_dataset_from_directory

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

DATASET_PATH=os.path.join(BASE_DIR,"dataset")
MODEL_PATH=os.path.join(BASE_DIR,"models","autoencoder_model.h5")

os.makedirs(os.path.join(BASE_DIR,"models"),exist_ok=True)

print("Loading dataset from local folder...")

dataset=image_dataset_from_directory(
DATASET_PATH,
image_size=(64,64),
batch_size=32,
label_mode=None
)

dataset=dataset.map(lambda x:x/255.0)
dataset=dataset.map(lambda x:(x,x))

print("Building Autoencoder...")

encoder=models.Sequential([
layers.Input(shape=(64,64,3)),
layers.Conv2D(32,3,activation='relu',padding='same'),
layers.MaxPooling2D(2),
layers.Conv2D(64,3,activation='relu',padding='same'),
layers.MaxPooling2D(2)
])

decoder=models.Sequential([
layers.Conv2DTranspose(64,3,strides=2,activation='relu',padding='same'),
layers.Conv2DTranspose(32,3,strides=2,activation='relu',padding='same'),
layers.Conv2D(3,3,activation='sigmoid',padding='same')
])

autoencoder=models.Sequential([encoder,decoder])

autoencoder.compile(
optimizer="adam",
loss="mse"
)

print("Training Autoencoder...")

autoencoder.fit(dataset,epochs=3)
autoencoder.save(MODEL_PATH)

print("Autoencoder saved successfully.")