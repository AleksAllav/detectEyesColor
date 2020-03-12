# import the necessary packages
import cv2
import numpy as np
from lib import detectEyesColorLib
    
class Face():    
    def __init__(self, image):
        self.image = image
        
        # Convert to grayscale
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @property
    def face(self):
        face = detectFaces(self.image, self.gray)
        #print(faces)
        return face
    
    @property
    def scinColor(self):
        scinColor = detectEyesColorLib.getDominantColor(self.image, k=4)
        return scinColor
    
    @property
    def eyes(self):
        eyes = detectEyes(self.image, self.gray, self.face)
        return eyes
        
    @property
    def eyesIrises(self):
        irises = []
        for eye in self.eyes:
            irises.append(detectIrises(eye, self.scinColor))
        return irises
            
            
def detectFaces(clone, gray):
    ''' 
    This function gets faces images,
    detects faces to crop the image to speed up eye search.     
    If face isn't detected, the function returns original image shape.
    
    Arguments:
    
    Returns:  
    Faces coordinates
    
    '''
       
    # Load haarcascades
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    # Find faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 8,
        minSize = (20, 20)
    )
    
    # Set face all image if face was not detected 
    if len(faces)==0:
        faces = [[0, 0, clone.shape[1], clone.shape[0]]]
        
    return faces
        
        
def detectEyes(image, gray, faces):
    '''
    This function detects eyes.   
    
    Arguments:
    
    Returns:  
    eyes images names
    
    '''
    
    # Load haarcascades
    eyeCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    
    # Find eyes
    eyesImages = []
    eyesImagesNames = []
    j=1
    for (x, y, w, h) in faces:
        # Crop the area with faces
        roi_gray = gray[y : y + h, x : x + w] 
        roi_color = image[y : y + h, x : x + w]   
        
        # Detect eyes
        eyes = eyeCascade.detectMultiScale(
            roi_gray,              
            scaleFactor = 1.2,       
            minNeighbors = 4,
            minSize = (150, 150)
        )  
        
        # Save images for transfer
        for (ex, ey, ew, eh) in eyes: 
            eye = roi_color[ey : ey + eh, ex : ex + ew]
            eyesImages.append(eye)
                    
    return eyesImages


def detectIrises(eye, scinColor):
    ''' 
    This function gets eyes images, then detects irises on processed images  
    and returns irises which cropped by mask.  
    
    Arguments:
    
    Returns:
    
    '''       

    clone = eye.copy()
    
    # Get processed images
    images, imagesName = processImage(clone)
    
    # Find irises and write them
    irisesImages = []
    irisesImagesNames = []
    for image in images:
        # Find iris on current image of eye
        irises = findCirclesByMask(eye.copy(), image, scinColor)
        
        # TODO: implement finding iris in circles
        # iris = findIris(circles)
        
        for j, iris in enumerate(irises):
            if iris.size != 0:
                # Save images for transfer
                irisesImages.append(iris)
                # Debug for manual checking: save images of eyes
                #cv2.imwrite("./labeled/detectEye/" + name+ "_" + imagesName[i] + "_iris" + str(j) + ".jpg", iris)
                #irisesImagesNames.append(name+ "_" + imagesName[i] + "_iris" + str(j))
        
    return irisesImages       

def processImage(clone):
    # Use bilateralFilter and convert to grayscale
    image = cv2.bilateralFilter(clone, 10, 100, 100)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return [grey], ['grey']

def findCirclesByMask(image, changed, scinColor):
        
    # Find circles 
    rows  = changed.shape[0]
    circles = cv2.HoughCircles(changed, cv2.HOUGH_GRADIENT, 1, rows/8,
                               param1 = 100, param2 = 30,
                               minRadius = 10, maxRadius = 100)
        
    crop = []    
    # Find all circles with using mask
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, radius in circles[0, :]:
            center = (x, y)
            # Save original image
            clone = image.copy()
            
            # Create mask
            height,width,_ = clone.shape
            mask = np.zeros((height, width), np.uint8)
            
            
            # Draw on mask
            cv2.circle(mask, center, radius,(255, 255, 255), thickness=-1)
            
            # Copy that image using that mask
            masked_data = cv2.bitwise_and(clone, clone, mask=mask)

            # Apply Threshold
            _,thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i, contour in enumerate(contours):
                # Get the bounding rect
                x, y, w, h = cv2.boundingRect(contour)
                # Crop masked_data
                iris = masked_data[y : y + h, x : x + w]
                crop.append(iris)
                dom_color = detectEyesColorLib.getDominantColor(iris,5)
                dom_color_hsv = np.full(iris.shape, dom_color, dtype='uint8')
                dom_color_bgr = cv2.cvtColor(dom_color_hsv, cv2.COLOR_HSV2BGR)
                output_image = np.hstack((iris, dom_color_bgr))
                cv2.imwrite('./labeled/detectEyeColor/' + str(x) + str(y) + '_dominantEyeColor.jpg', output_image)
    return crop