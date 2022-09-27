import matplotlib.pyplot as plt
import numpy as np
import time

from encoder import encoderModel
from image_loader import training_data_x, training_data_y, test_data_x, test_data_y, train_source_generator, train_target_generator

encoderModel.load_weights('weights/encoder.h5')


encoded_images = encoderModel.predict(test_data_x)

for i in range(10):
	fig, (ax1, ax2, ax3) = plt.subplots(3)

	ax1.imshow(test_data_x[i])
	ax2.imshow(test_data_y[i])
	ax3.imshow(encoded_images[i])

	plt.savefig('output/result_prec_%s_%s.png' % (i, time.time()))

encoded_images = encoderModel.predict(training_data_x)
for i in range(10):
	fig, (ax1, ax2, ax3) = plt.subplots(3)

	ax1.imshow(training_data_x[i])
	ax2.imshow(training_data_y[i])
	ax3.imshow(encoded_images[i])

	plt.savefig('output/result_train_%s_%s.png' % (i, time.time()))
