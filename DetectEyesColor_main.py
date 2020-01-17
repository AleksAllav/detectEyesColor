# import the necessary packages
import lib.detectEyesLib
import lib.detectIrisesLib
import lib.detectEyesColorLib
import cv2

       
if __name__ == "__main__":
    
    # load images of faces
    facesImages = ['face1','face2','face3','face4','face9','face8','face7']

    # detect eyes on images of faces and return images of eyes
    eyesImages = []
    for name in facesImages:
        eyesImages.append(lib.detectEyesLib.findEyes(name, cv2.imread('./pictures/faces/' + name + '.jpg')))
    
    # debug 
    # eyesImages= ['eye1','eye2','eye3','eye4','eye5','eye6','eye7','eye8','eye9','eye10']
    
    # load the image and return countours of irises 
    irisesImages = []
    for _ in eyesImages:        
        for name in _:
            irisesImages.append(lib.detectIrisesLib.findIrises(name, cv2.imread('./labeled/detectFace/'+name+'.jpg')))
            # debug
            # DetectEye_findIrises.writeFindedCountoursOnEyes(name, cv2.imread('./labeled/detectFace/'+name+'.jpg'))               
    
    # load the image of irises and the detect color of eye
    for _ in irisesImages:        
        for name in _:         
            lib.detectEyesColorLib.detectEyesColor(name, cv2.imread('./labeled/detectEye/'+name+'.jpg'))    
    