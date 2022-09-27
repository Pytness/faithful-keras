#!/usr/bin/env python3

from encoder import encoderModel
import matplotlib.pyplot as plt
import numpy as np
# import random

from image_loader import training_data_x, training_data_y, train_source_generator, train_target_generator

# print('Loading images...')

# for filename in glob.glob('extracted_images/minecraft_1.16.1/*.png'):
# 	img = img_to_array(load_img(filename, color_mode='rgba'))
# 	if img.shape != (16, 16, 4):
# 		print(img.shape, filename)

encoderModel.fit(
    training_data_x, training_data_y,
	epochs=100,
	batch_size=8,
	steps_per_epoch=len(training_data_x) // 8,
	shuffle=False
)

encoderModel.save_weights("weights/encoder.h5")
