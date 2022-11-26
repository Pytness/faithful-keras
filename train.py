import tensorflow as tf

# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from model import model
from image_loader import training_data_x, training_data_y, datagen_generator 

def ssim_loss(y_true, y_pred):
	return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1, filter_size=5))

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath='./checkpoints/chkpt',
	save_weights_only=True,
	monitor='loss',
	mode='min',
    save_freq=20)

cosine_decay_learning_rate = tf.keras.callbacks.LearningRateScheduler(
    tf.keras.optimizers.schedules.CosineDecay(0.0019, 5000))

model.compile(optimizer='adam', loss=ssim_loss)
model.fit(
    training_data_x, training_data_y,
	epochs=5000,
	batch_size=256,
    callbacks=[model_checkpoint_callback, cosine_decay_learning_rate],
)

model.save_weights("weights/model.h5")
