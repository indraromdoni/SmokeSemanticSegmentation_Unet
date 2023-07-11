import cv2
import tensorflow as tf
import numpy as np

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

model = tf.keras.models.load_model("smoke-detector.h5")

cap = cv2.VideoCapture("c:\\Users\\Anindra\\Downloads\\Foundry worker puts wet scrap metal in furnace.mp4")
while True:
    _, frame = cap.read()
    r_img = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
    inp = r_img[tf.newaxis, ...]
    pred = model.predict(inp)[0]
    hasil = np.array(tf.keras.utils.array_to_img(pred))
    white = np.sum(hasil >= 125)
    black = np.sum(hasil == 0)
    total = white + black
    pers = round((white/total)*100)
    hasil = cv2.cvtColor(hasil, cv2.COLOR_GRAY2BGR)
    cv2.putText(hasil, str(pers)+"%", (5,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
    conc = np.hstack((r_img, hasil))
    cv2.imshow("Hasil", conc)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()