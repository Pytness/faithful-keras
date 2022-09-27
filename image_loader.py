from keras_preprocessing.image import img_to_array, load_img, ImageDataGenerator
import numpy as np
import glob


datagen_args = {
	'horizontal_flip': True,
	'vertical_flip': True,
	# 'channel_shift_range': 4,
    'rotation_range': 90,
    'width_shift_range': .5,
    'height_shift_range': .5,
	'fill_mode': 'constant',
	'cval': 0,
	'dtype': 'float32'
}

source_datagen = ImageDataGenerator(**datagen_args)
target_datagen = ImageDataGenerator(**datagen_args)

training_data_x = np.asarray([
	img_to_array(load_img(filename, color_mode='rgba'), dtype='float32').tolist()
	for filename in glob.glob('extracted_images/minecraft_1.16.1/*.png')
])
training_data_x = training_data_x.astype('float32') / 255
print(training_data_x.shape)
training_data_y = np.asarray([
	img_to_array(load_img(filename, color_mode='rgba'), dtype='float32').tolist()
	for filename in glob.glob('extracted_images/faithful_1.16.1/*.png')
])
training_data_y = training_data_y.astype('float32') / 255


test_data_x = training_data_x[-64:]
training_data_x = training_data_x[:-64]

test_data_y = training_data_y[-64:]
training_data_y = training_data_y[:-64]

datagen = ImageDataGenerator(**datagen_args)
datagen.fit(training_data_x)
 

train_source_generator = datagen.flow(training_data_x, training_data_y, batch_size=32, seed=1)
train_target_generator = source_datagen.flow(training_data_y, batch_size=32, seed=1)
