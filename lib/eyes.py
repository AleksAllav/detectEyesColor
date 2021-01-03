# import the necessary packages
import cv2
from .eyes_color import get_dominant_color


def detect_eyes(name, image):
    """
    This function detects eyes.

    Arguments:

    Returns:
    eyes images names

    """
    
    # Load haarcascades
    eyeCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    
    # Save original image
    clone = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
    
    faces = detect_faces(clone, gray)
    
    # Debug: Draw rectangle around faces    
    for (x, y, w, h) in faces:
        cv2.rectangle(clone, (x, y), (x+w, y+h), (255, 255, 0), 2)  

    # Find eyes
    eyesImages = []
    eyesImagesNames = []
    j = 1
    for (x, y, w, h) in faces:
        # Crop the area with faces
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = image[y: y + h, x: x + w]

        # Detect eyes
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(150, 150)
        )

        # Find scin color 
        scinColor = get_dominant_color(roi_color, k=4)
        
        # Draw the area with eyes
        for (ex, ey, ew, eh) in eyes:
            eye = roi_color[ey: ey + eh, ex: ex + ew]
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                       
            # Save images for transfer
            eyesImages.append(eye)
            
            # Debug for manual checking: save images of eyes
            #eyesImagesNames.append(name + '_eye' + str(j))
            #cv2.imwrite('./labeled/detectFace/' + name + '_eye' + str(j) + '.jpg', eye)

        
        # Debug for manual checking: save labeled face and eyes
        cv2.imwrite('./labeled/detectFace/' + name + 'Labeled.jpg', roi_color)
        
    return scinColor, eyesImages


def detect_faces(clone, gray):
    """
    This function gets faces images,
    detects faces to crop the image to speed up eye search.
    If face isn't detected, the function returns original image shape.

    Arguments:

    Returns:
    Faces coordinates

    """
    
    # Load haarcascades
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    # Find faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=8,
        minSize=(20, 20)
    )
    
    # Set face all image if face was not detected 
    if len(faces) == 0:
        faces = [[0, 0, clone.shape[1], clone.shape[0]]]
    
    # Debug
    #faces_detected = "Лиц обнаружено: " + format(len(faces))
    #print(faces_detected)
    
    return faces
