import matplotlib.pyplot as plt
import numpy as np
import time

from encoder import encoderModel
from image_loader import training_data_x, training_data_y, test_data_x, test_data_y 

encoderModel.load_weights('weights/encoder.h5')


encoded_images = encoderModel.predict(test_data_x)


# plot 10 collumns of 3 images each
columns = 10
rows = 3

fig, axs = plt.subplots(rows, columns, figsize=(20, 4))

for i in range(columns):

	axs[0, i].imshow(test_data_x[i])
	axs[1, i].imshow(test_data_y[i])
	axs[2, i].imshow(encoded_images[i])
plt.savefig('output/result_test_%s_%s.png' % (i, time.time()))

encoded_images = encoderModel.predict(training_data_x)

fig, axs = plt.subplots(rows, columns, figsize=(20, 4))

for i in range(columns):

	axs[0, i].imshow(training_data_x[i])
	axs[1, i].imshow(training_data_y[i])
	axs[2, i].imshow(encoded_images[i])
plt.savefig('output/result_train_%s_%s.png' % (i, time.time()))

