# import the necessary packages
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import cv2


def detectEyesColor(name, image):
    ''' 
    This function gets irises images, than gets dominant color on image,
    writes image with adding the area which filled by dominant color .        
    
    Arguments:
    
    Returns:    
    
    '''
    # TODO: add sorting irises by color and pass by wrong detected irises
    if image.size != 0:            
        iris_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dom_color = getDominantColor(iris_hsv)
        dom_color_hsv = np.full(image.shape, dom_color, dtype='uint8')
        dom_color_bgr = cv2.cvtColor(dom_color_hsv, cv2.COLOR_HSV2BGR)
        output_image = np.hstack((image, dom_color_bgr))
        cv2.imwrite('./labeled/detectEyeColor/' + name + '_dominantEyeColor.jpg', output_image)


def getDominantColor(image, k=4):
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)
    label_counts = Counter(labels)
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    return list(dominant_color)
