# import the necessary packages
import pathlib

import cv2
from lib import eyes, eyes_color, irises

if __name__ == '__main__':
    
    # Load images of faces
    faces_images = list(pathlib.Path('./pictures/faces/').glob('*.jpg'))

    # Detect eyes on images of faces and return images of eyes
    eyesImages = []
    for image in faces_images:
        name = image.resolve().stem
        skinColor, currentEyesImages = eyes.detect_eyes(name, cv2.imread(str(image)))
        eyesImages.append(currentEyesImages)
    print('Debug: The end of detecting eyes')
    
    # Load the image and return contours of irises
    irisesImages = []
    for current_face_eyes in eyesImages:
        for eye in current_face_eyes:
            irisesImages.append(irises.detect_irises(name, eye))
    print('Debug: The end of detecting irises')
    
    # Load the image of irises and the detected color of eye
    for i, current_face_irises in enumerate(irisesImages):
        for j, iris in enumerate(current_face_irises):
            output_image = eyes_color.detect_eyes_color(name, iris)
            cv2.imwrite('./pictures/labeled/detectedEyeColor_' + str(i) + str(j) + '_dominantEyeColor.jpg', output_image)
    print('Debug: The end of detecting eyes color')
