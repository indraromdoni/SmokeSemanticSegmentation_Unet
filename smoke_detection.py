import cv2
import tensorflow as tf
import numpy as np

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

model = tf.keras.models.load_model("smoke-detector.h5")
img = cv2.imread("Smoke-detector-4/train/mf30t295_jpg.rf.59d1c83ddeeb3bbfc5b3858e0ec5a40a.jpg")
r_img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
inp = r_img[tf.newaxis, ...]
print(r_img.shape, inp.shape)
pred = model.predict(inp)[0]
hasil = np.array(tf.keras.utils.array_to_img(pred))
hasil = cv2.cvtColor(hasil, cv2.COLOR_GRAY2BGR)
conc = np.hstack((r_img, hasil))
cv2.imshow("Hasil", conc)
cv2.waitKey(0)
cv2.destroyAllWindows()