#!/usr/bin/env python3

from encoder import encoderModel
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import random

from image_loader import training_data_x, training_data_y, datagen_generator 

# print('Loading images...')

# for filename in glob.glob('extracted_images/minecraft_1.16.1/*.png'):
# 	img = img_to_array(load_img(filename, color_mode='rgba'))
# 	if img.shape != (16, 16, 4):
# 		print(img.shape, filename)



# print('Creating model...')
# data_x, data_y = [ next(datagen_generator) for _ in range(10000) ]

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath='./checkpoint',
	save_weights_only=True,
	monitor='accuracy',
	mode='max',
	save_best_only=True)

encoderModel.fit(
    training_data_x, training_data_y,
	epochs=10000,
	batch_size=len(training_data_x),
    callbacks=[model_checkpoint_callback],
	# steps_per_epoch=len(training_data_x) // 8,
)

encoderModel.save_weights("weights/encoder.h5")
