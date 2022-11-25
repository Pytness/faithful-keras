#!/usr/bin/env python3

from keras.models import Sequential, Model
from keras.layers import Activation, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Reshape
import tensorflow as tf
import keras.metrics
image_shape = (16, 16, 4)
# Define the model

# 1st encoder convolution layer
encoderInput = Input(shape=image_shape, name="encoderInput")

# -----------------------------------------------------------------
encoderLayer = Conv2D(16, (3, 3), activation="relu")(encoderInput)
encoderLayer = Conv2D(32, (3, 3), activation="relu")(encoderLayer)
encoderLayer = Conv2D(32, (3, 3), activation="relu")(encoderLayer)
encoderLayer = Conv2D(64, (3, 3), activation="relu")(encoderLayer)
encoderLayer = Conv2D(64, (3, 3), activation="relu")(encoderLayer)
encoderLayer = Conv2D(256, (3, 3), activation="relu")(encoderLayer)
encoderLayer = Conv2D(256, (1, 1), activation="relu")(encoderLayer)

encoderLayer = Conv2D(256, (3, 3), activation="relu", padding="same")(encoderLayer)
encoderLayer = Conv2D(256, (3, 3), activation="relu", padding="same")(encoderLayer)
encoderLayer = Conv2D(256, (3, 3), activation="relu", padding="same")(encoderLayer)
encoderLayer = Conv2D(256, (3, 3), activation="relu", padding="same")(encoderLayer)
encoderLayer = Conv2D(256, (3, 3), activation="relu", padding="same")(encoderLayer)
encoderLayer = Conv2D(256, (3, 3), activation="relu", padding="same")(encoderLayer)



encoderLayer = Reshape((32, 32, 4))(encoderLayer)
encoderLayer = Activation("sigmoid")(encoderLayer)

encoderModel = Model(encoderInput, encoderLayer)
encoderModel.summary()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# encoderModel.summary()
# Loss functtion
def ssim_loss(y_true, y_pred):
	return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

encoderModel.compile(
	optimizer='adam',
	loss=ssim_loss,
	metrics=['accuracy'],
)

# save models
with open('models/encoder.json', 'w') as f:
	f.write(encoderModel.to_json())
