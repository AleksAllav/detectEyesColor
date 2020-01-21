# import the necessary packages
import cv2

def detectEyes(name, image):
    ''' 
    This function gets faces images,
    detects faces to crop the image to speed up eye search,
    than detects eyes and returns them.        
    
    Arguments:
    
    Returns:
    
    
    '''
    
    # load haarcascades
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    
    # save original image
    clone = image.copy()
    
    # convert to grayscale
    gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

    # find faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor= 1.2,
        minNeighbors= 8,
        minSize=(20, 20)
    )
    # set face all image if face was not detected 
    if len(faces)==0:
        faces = [[0,0, clone.shape[1], clone.shape[0]]]
    print(faces)
    
    # debug
    #faces_detected = "Лиц обнаружено: " + format(len(faces))
    #print(faces_detected)

    # draw rectangle around faces    
    for (x, y, w, h) in faces:
        cv2.rectangle(clone, (x, y), (x+w, y+h), (255, 255, 0), 2)  

    # find eyes
    eyesImages = []
    j=1
    for (x, y, w, h) in faces:
        # crop the area with faces
        roi_gray = gray[y:y + h, x:x + w] 
        roi_color = image[y:y + h, x:x + w]   
        # find eyes
        eyes = eyeCascade.detectMultiScale(
            roi_gray,              
            scaleFactor=1.2,       
            minNeighbors=4,
            minSize=(150, 150),
        )        
        # draw the area with eyes
        for (ex, ey, ew, eh) in eyes:
            img = roi_color[ey:ey + eh, ex:ex + ew]   
            eyesImages.append(name + '_eye' + str(j))
            cv2.imwrite('./labeled/detectFace/' + name + '_eye' + str(j) + '.jpg', img)
            j += 1
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  
        cv2.imwrite('./labeled/detectFace/' + name + 'Labeled.jpg', roi_color)
        
    return eyesImages

    