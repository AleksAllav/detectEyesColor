# import the necessary packages
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import cv2
    

def get_dominant_color(image, k=4):
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)
    label_counts = Counter(labels)
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    return list(dominant_color)
        
def detectEyesColor(name, image):
    #irises = findCirclesAndReturnIris(resized.copy(), image)
    iris = image
#    for iris in irises:
    if iris.size != 0:            
        iris_hsv = cv2.cvtColor(iris, cv2.COLOR_BGR2HSV)
        dom_color = get_dominant_color(iris_hsv)
        dom_color_hsv = np.full(iris.shape, dom_color, dtype='uint8')
        dom_color_bgr = cv2.cvtColor(dom_color_hsv, cv2.COLOR_HSV2BGR)
        output_image = np.hstack((iris, dom_color_bgr))
        cv2.imwrite('./labeled/detectEyeColor/' + name + '_dominantEyeColor.jpg', output_image)