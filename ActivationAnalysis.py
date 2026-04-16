import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

IMG=224

def preprocess(image,label):
	image=tf.image.resize(image,(IMG,IMG))
	image=image/255.0
	return image,label

train,test=tfds.load(
"eurosat",
split=["train[:80%]","train[80%:]"],
as_supervised=True
)

train=train.map(preprocess).batch(32)
test=test.map(preprocess).batch(32)

def build_model(act):
	model=tf.keras.Sequential([
	tf.keras.layers.Conv2D(32,3,activation=act,input_shape=(224,224,3)),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(64,activation=act),
	tf.keras.layers.Dense(10,activation="softmax")
	])
	model.compile(
	optimizer="adam",
	loss="sparse_categorical_crossentropy",
	metrics=["accuracy"]
	)
	return model

relu_model=build_model("relu")
relu_model.fit(train,epochs=3,validation_data=test)
plt.plot(relu_model.history.history["accuracy"])
plt.title("Activation Function Performance")
plt.show()