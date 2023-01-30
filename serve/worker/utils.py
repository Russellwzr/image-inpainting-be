import cv2
import numpy as np


def read_image_from_binary(byte_obj, color=cv2.IMREAD_COLOR):
    img = np.frombuffer(byte_obj, dtype='uint8')
    img = cv2.imdecode(img, color)
    return img


def write_image_to_binary(img):
    img = cv2.imencode('.jpg', img)
    assert img[0]
    byte_obj = img[1].tobytes()
    return byte_obj
