import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models", "vgg16_model.keras")

# 1. Uniform Augmentation Logic
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])

train_ds = image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training", seed=123,
    image_size=(224, 224), batch_size=32
).map(lambda x, y: (augmentation(x, training=True), y))

val_ds = image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="validation", seed=123,
    image_size=(224, 224), batch_size=32
)

# 2. Build Fine-Tuned Model (Identical to Colab)
base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:-4]: # Keep last block trainable
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(len(train_ds.class_names), activation="softmax")(x)

model = models.Model(base_model.input, output)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Starting local test training...")
model.fit(train_ds, validation_data=val_ds, epochs=10)
model.save(MODEL_PATH)