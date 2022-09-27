#!/usr/bin/env python3

from keras.models import Sequential, Model
from keras.layers import Activation, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import keras.metrics
image_shape = (16, 16, 4)
# Define the model

# 1st encoder convolution layer
encoderInput = Input(shape=image_shape, name="encoderInput")

# -----------------------------------------------------------------
encoderLayer = Conv2D(8, (2, 2), padding='same')(encoderInput)
encoderLayer = Activation('relu')(encoderLayer)
encoderLayer = Conv2D(8, (2, 2), padding='same')(encoderInput)
encoderLayer = Activation('relu')(encoderLayer)
encoderLayer = UpSampling2D(2)(encoderLayer)

# -----------------------------------------------------------------
encoderLayer = Conv2D(12, (2, 2), padding='same')(encoderLayer)
encoderLayer = Activation('relu')(encoderLayer)
encoderLayer = UpSampling2D(2)(encoderLayer)

# -----------------------------------------------------------------
encoderLayer = Conv2D(16, (8, 8), padding='same')(encoderLayer)
encoderLayer = Activation('relu')(encoderLayer)
encoderLayer = UpSampling2D(2)(encoderLayer)

# -----------------------------------------------------------------


# -----------------------------------------------------------------
encoderLayer = Conv2D(12, (8, 8), padding='same')(encoderLayer)
encoderLayer = Activation('relu')(encoderLayer)
encoderLayer = Conv2D(12, (8, 8), padding='same')(encoderLayer)
encoderLayer = Activation('relu')(encoderLayer)
encoderLayer = Conv2D(12, (8, 8), padding='same')(encoderLayer)
encoderLayer = Activation('relu')(encoderLayer)
encoderLayer = Conv2D(12, (8, 8), padding='same')(encoderLayer)
encoderLayer = Activation('relu')(encoderLayer)

encoderLayer = MaxPooling2D(pool_size=(2, 2), padding='same')(encoderLayer)

encoderLayer = Conv2D(12, (2, 2), padding='same')(encoderLayer)
encoderLayer = Activation('relu')(encoderLayer)
encoderLayer = MaxPooling2D(pool_size=(2, 2), padding='same')(encoderLayer)

# 2nd encoder convolution layer
encoderLayer = Conv2D(8, (2, 2), padding='same')(encoderLayer)
encoderLayer = Activation('relu')(encoderLayer)
encoderLayer = Conv2D(4, (8, 8), padding='same')(encoderLayer)
encoderLayer = Activation('relu')(encoderLayer)
encoderLayer = Conv2D(4, (8, 8), padding='same')(encoderLayer)
encoderLayer = Activation('relu')(encoderLayer)
# encoderLayer = MaxPooling2D(pool_size=(2,2), padding='same')(encoderLayer)

# 2nd encoder convolution layer again
# encoderLayer = Conv2D(3,(3, 3), padding='same')(encoderLayer)
encoderLayer = Activation('relu', name="encoderOutput")(encoderLayer)

encoderModel = Model(encoderInput, encoderLayer)
encoderModel.summary()

# encoderModel.summary()


encoderModel.compile(
	optimizer='adam',
	loss='mean_squared_error',
	metrics=['accuracy']
)

# save models
with open('models/encoder.json', 'w') as f:
	f.write(encoderModel.to_json())
