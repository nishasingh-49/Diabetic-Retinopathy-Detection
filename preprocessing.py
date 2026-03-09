import cv2
import numpy as np
def crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > 10
    return img[np.ix_(mask.any(1), mask.any(0))]
def preprocess(img, size=224):
    img = crop_black(img)
    img = cv2.resize(img, (size, size))
    img = img / 255.0
    return img