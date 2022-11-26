import matplotlib.pyplot as plt
import numpy as np
import sys

from model import model

# model.load_weights('weights/model.h5')
model.load_weights('checkpoints/chkpt')

image_to_infer = sys.argv[1]

image = plt.imread(image_to_infer, format="PNG")
image = image.reshape((1, 16, 16, 4))
predict = model.predict(image)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 4))
ax1.imshow(image[0])
ax2.imshow(predict[0])
plt.savefig('inference_result.png')