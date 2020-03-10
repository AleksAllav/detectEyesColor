# import the necessary packages
import cv2
import lib.detectEyesColorLib
    
class Face():
    # Load haarcascades
    eyeCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    
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
    def faceScinColor(self):
        scinColor = lib.detectEyesColorLib.getDominantColor(self.image, k=4)
        return scinColor
    
    @property
    def eyes(self):
        #face, eyes, scinColor
        eyes = detectEyes(self.image, self.gray, self.face)
        
        
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

        