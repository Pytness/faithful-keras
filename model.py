import tensorflow as tf
from tensorflow import keras

def deconv_block(input, filters, kernel_size, padding="same"):
    layer = keras.layers.Conv2DTranspose(filters, kernel_size, padding=padding)(input)
    # layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.activations.relu(layer)
    return layer

input = keras.layers.Input(shape=(16, 16, 4))
layer = deconv_block(input, 256, (3, 3), padding="valid")
layer = deconv_block(layer, 256, (3, 3), padding="valid")
layer = deconv_block(layer, 256, (3, 3), padding="valid")
layer = deconv_block(layer, 256, (3, 3), padding="valid")
layer = deconv_block(layer, 256, (3, 3))
layer = deconv_block(layer, 256, (3, 3))
layer = deconv_block(layer, 256, (3, 3))
layer = deconv_block(layer, 256, (3, 3), padding="valid") #upsample
layer = deconv_block(layer, 128, (3, 3))
layer = deconv_block(layer, 128, (3, 3))
layer = deconv_block(layer, 128, (3, 3))
layer = deconv_block(layer, 128, (3, 3))
layer = deconv_block(layer, 128, (3, 3))
layer = deconv_block(layer, 128, (3, 3), padding="valid") #upsample
layer = deconv_block(layer, 64, (3, 3))
layer = deconv_block(layer, 64, (3, 3))
layer = deconv_block(layer, 64, (3, 3))
layer = deconv_block(layer, 64, (3, 3), padding="valid") #upsample
layer = deconv_block(layer, 32, (3, 3))
layer = deconv_block(layer, 32, (3, 3))
layer = deconv_block(layer, 32, (3, 3), padding="valid") #upsample
layer = deconv_block(layer, 4, (3, 3))
layer = keras.layers.Conv2DTranspose(4, (5, 5), activation="sigmoid", padding="same")(layer)

model = keras.Model(inputs=input, outputs=layer)
print(model.summary())
