import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog

def extract_color_hist(img):
    img = cv2.resize(img, (64, 64))
    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float") / (hist.sum() + 1e-6)
    return hist

def extract_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    features = hog(gray, orientations=6, pixels_per_cell=(16,16),
                   cells_per_block=(2,2), block_norm='L2-Hys', visualize=False)
    return features

'''
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    return np.hstack([extract_color_hist(img), extract_lbp(img), extract_hog(img)])
'''

def extract_features(image_or_path):
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path)
    else:
        img = image_or_path
    if img is None:
        return None
    return np.hstack([extract_color_hist(img), extract_lbp(img), extract_hog(img)])
