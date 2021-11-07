from tensorflow.keras.models import load_model
import cv2
import numpy as np

IMG_SIZE = 224
my_model = load_model('models/model_ckpt.h5')

def predict(path):
    label_map = {
        0: 'good',
        1: 'poor'
    }

    npimg = np.fromstring(path, np.uint8)
    test_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    test_image = cv2.resize(test_image, (IMG_SIZE, IMG_SIZE))
    test_image = np.expand_dims(test_image, axis=0)

    pred = my_model.predict(test_image)
    print(f'\nModel prediction probabilities: {pred}')
    pred = np.argmax(pred)
    pred = label_map[pred]
    return pred