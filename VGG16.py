import tensorflow as tf
from tensorflow.keras import layers, models
import os

# CONFIG
IMG_SIZE = (512, 512)
BATCH_SIZE = 4  # Lower batch for local VS Code stability
DATA_PATH = "./dataset_extracted" 

# DATA LOADING
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
).map(lambda x, y: (x/255.0, y)).prefetch(tf.data.AUTOTUNE)

# ACTUAL VGG16 ARCHITECTURE
def build_vgg16_classifier():
    # Load the actual VGG16 model weights from ImageNet
    base_model = tf.keras.applications.VGG16(
        weights='imagenet', 
        include_top=False, 
        input_shape=(512, 512, 3)
    )
    base_model.trainable = False # Keep the expert features frozen

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4), # Prevents overfitting
        layers.Dense(10, activation='softmax') # 10 Satellite Classes
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_vgg16_classifier()
print("Starting VGG16 Training...")
model.fit(train_ds, epochs=15)
model.save("models/vgg16_model.keras")