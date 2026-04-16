import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing import image_dataset_from_directory

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

DATASET_PATH=os.path.join(BASE_DIR,"dataset")
MODEL_PATH=os.path.join(BASE_DIR,"models","vgg16_model.h5")

print("Loading dataset...")

train_ds=image_dataset_from_directory(
DATASET_PATH,
validation_split=0.2,
subset="training",
seed=123,
image_size=(224,224),
batch_size=32
)

val_ds=image_dataset_from_directory(
DATASET_PATH,
validation_split=0.2,
subset="validation",
seed=123,
image_size=(224,224),
batch_size=32
)

class_names=train_ds.class_names

print("Building VGG16 model...")

base_model=VGG16(
weights="imagenet",
include_top=False,
input_shape=(224,224,3)
)

base_model.trainable=False

x=layers.Flatten()(base_model.output)
x=layers.Dense(256,activation="relu")(x)
x=layers.Dense(len(class_names),activation="softmax")(x)

model=models.Model(base_model.input,x)

model.compile(
optimizer="adam",
loss="sparse_categorical_crossentropy",
metrics=["accuracy"]
)

print("Training classifier...")

model.fit(train_ds,validation_data=val_ds,epochs=5)

if not os.path.exists("models"):
	os.makedirs("models")

model.save(MODEL_PATH)

print("VGG16 model saved successfully.")