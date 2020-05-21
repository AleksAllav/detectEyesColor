# import the necessary packages
import cv2
from lib import (
    detectEyesLib,
    detectIrisesLib,
    detectEyesColorLib,
)


if __name__ == '__main__':
    
    # Load images of faces
    facesImages = ('face1', 'face2', 'face3')

    # Detect eyes on images of faces and return images of eyes
    eyesImages = []
    for name in facesImages:
        skinColor, currentEyesImages = detectEyesLib.detectEyes(name, cv2.imread('./pictures/faces/' + name + '.jpg'))
        eyesImages.append(currentEyesImages)
    print('Debug: The end of detecting eyes')
    
    # Debug 
    # eyesImages= ['eye1','eye2','eye3','eye4','eye5','eye6','eye7','eye8','eye9','eye10']
    
    # Load the image and return countours of irises 
    irisesImages = []
    for _ in eyesImages: 
        for eye in _:
            irisesImages.append(detectIrisesLib.detectIrises(name, eye))
    print('Debug: The end of detecting irises')
    
    # Load the image of irises and the detect color of eye
    for _ in irisesImages:
        for iris in _:
            detectEyesColorLib.detectEyesColor(name, iris)
    print('Debug: The end of detecting eyes color')
