# import the necessary packages
import numpy as np
import cv2


def detectIrises(name, image, scinColor = None):
    ''' 
    This function gets eyes images, then detects irises on processed images  
    and returns irises which cropped by mask.  
    
    Arguments:
    
    Returns:
    
    '''       
    
    clone = image.copy()
    
    # Get processed images
    images, imagesName = processImage(clone)
    
    # Find irises and write them
    irisesImages = []
    irisesImagesNames = []
    for i in range(len(images)):
        # Find iris on current image of eye
        irises = findCirclesByMask(image.copy(), images[i])
        
        # TODO: implement finding iris in circles
        # iris = findIris(circles)
        
        for j, iris in enumerate(irises):
            if iris.size != 0:
                # Save images for transfer
                irisesImages.append(iris)
                # Debug for manual checking: save images of eyes
                cv2.imwrite("./labeled/detectEye/" + name+ "_" + imagesName[i] + "_iris" + str(j) + ".jpg", iris)
                irisesImagesNames.append(name+ "_" + imagesName[i] + "_iris" + str(j))
        
    return irisesImages

def processImage(clone):
    # Use bilateralFilter and convert to grayscale
    image = cv2.bilateralFilter(clone, 10, 100, 100)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return [grey], ['grey']
 
def findCirclesByMask(image, changed):
        
    # Find circles 
    rows  = changed.shape[0]
    circles = cv2.HoughCircles(changed, cv2.HOUGH_GRADIENT, 1, rows/8,
                               param1 = 100, param2 = 30,
                               minRadius = 10, maxRadius = 100)
        
    crop = []    
    # Find all circles with using mask
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Save original image
            clone = image.copy()
            
            # Create mask
            height,width,_ = clone.shape
            mask = np.zeros((height, width), np.uint8)
            center = (i[0], i[1])
            
            # Draw on mask
            cv2.circle(mask, (i[0], i[1]), i[2],(255, 255, 255), thickness=-1)
            
            # Copy that image using that mask
            masked_data = cv2.bitwise_and(clone, clone, mask=mask)

            # Apply Threshold
            _,thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Get the bounding rect
                x, y, w, h = cv2.boundingRect(contour)
                # Crop masked_data
                crop.append(masked_data[y : y + h, x : x + w])
        
    return crop

# TODO: 
def findIris(circles):
    ''' 
    This function finds iris in circles.
    The function discards uncorrected circles by the color of the face and also function shouldn't count the color of the pupil.
    '''
    pass


# debug function
def findMethod(clone):    
    # Convert to grayscale
    gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    #thresh = cv2.bitwise_not(thresh)
    img = cv2.bitwise_not(thresh)

    # Use bilateralFilter and adaptiveThreshold
    image2 = cv2.bilateralFilter(clone, 10, 100, 100)
    grey = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    adthresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Threshold grayscaled image to get binary image
    ret,gray_threshed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Smooth an image
    bilateral_filtered_image = cv2.bilateralFilter(gray_threshed, 5, 175, 175)

    # Find edges
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
    
    images = [gray, blurred, thresh, img, grey, adthresh, bilateral_filtered_image, edge_detected_image] 
    imagesName = ['gray', 'blurred', 'thresh', 'img', 'grey', 'adthresh', 'bilateral_filtered_image', 'edge_detected_image'] 
    
    return images, imagesName

# Debug function
def findCountours(clone, changed):
    # Find all contours 
    contours, _ = cv2.findContours(changed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (50000 > area > 10000) ):
            contour_list.append(contour)
            
    # Draw contours
    cv2.drawContours(clone, contour_list, -1, (255, 0, 0), 5);    
    return clone

# Debug function
def findCircles(clone, changed):
    # Find circles 
    rows  = changed.shape[0]
    circles = cv2.HoughCircles(changed, cv2.HOUGH_GRADIENT, 1, rows/8,
                               param1 = 100, param2 = 30,
                               minRadius = 10, maxRadius = 100)
    # Draw circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # Circle center
            cv2.circle(clone, center, 1, (0, 100, 100), 3)
            # Circle outline
            radius = i[2]
            cv2.circle(clone, center, radius, (255, 0, 255), 3)
    return clone

# Debug function
def writeFindedCountoursOnEyes(name, image):
    # Resize it to a smaller factor so that
    # the shapes can be approximated better
    
    #resized = imutils.resize(image, width=300)
    #ratio = image.shape[0] / float(resized.shape[0])
    resized = image
    clone = resized.copy()

    images, imagesName = findMethod(clone)
    for i in range(len(images)):
        cv2.imwrite("./labeled/detectEye/" + name+ "_" + imagesName[i] + ".jpg", images[i])
        #imPath = "./labeled/detectEye/" + imagesName[i] + "_circling.jpg"    
        cv2.imwrite("./labeled/detectEye/" + name+ "_" + imagesName[i] + "_circling.jpg", findCircles(resized.copy(), images[i]))
        cv2.imwrite("./labeled/detectEye/" + name+ "_" + imagesName[i] + "_countering.jpg", findCountours(resized.copy(), images[i]))

