# import the necessary packages
import cv2
import lib.detectEyesLib
import lib.detectIrisesLib
import lib.detectEyesColorLib

       
if __name__ == '__main__':
    
    # Load images of faces
    facesImages = ['face1','face2','face3','face4','face5','face6','face7']

    # Detect eyes on images of faces and return images of eyes
    eyesImages = []
    for name in facesImages:
        scinColor, currentEyesImages = lib.detectEyesLib.detectEyes(name, cv2.imread('./pictures/faces/' + name + '.jpg'))
        eyesImages.append(currentEyesImages)
    print('Debug: The end of detecting eyes')
    
    # Debug 
    # eyesImages= ['eye1','eye2','eye3','eye4','eye5','eye6','eye7','eye8','eye9','eye10']
    
    # Load the image and return countours of irises 
    irisesImages = []
    for _ in eyesImages: 
        for eye in _:
            irisesImages.append(lib.detectIrisesLib.detectIrises(name, eye))            
    print('Debug: The end of detecting irises')
    
    # Load the image of irises and the detect color of eye
    for _ in irisesImages:
        for iris in _:
            lib.detectEyesColorLib.detectEyesColor(name, iris)
    print('Debug: The end of detecting eyes color')