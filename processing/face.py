# import the necessary packages
from collections import Counter

import cv2
import numpy as np
from lib import eyes_color
from sklearn.cluster import KMeans


class Face:
    def __init__(self, image):
        self.image = image
        # Convert to grayscale
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.skin_color = None

    @property
    def face(self):
        print('Face detecting started')
        face = self.detect_faces(clone=self.image, gray=self.gray)
        return face

    # TODO: refactor decorators 'property'
    # @property
    def skin_color(self):
        print('Skin color detecting started')
        self.skin_color = eyes_color.get_dominant_color(self.image)
        # return skin_color

    @property
    def eyes(self):
        print('Eyes detecting started')
        eyes = self.detect_eyes(self.image, self.gray, self.face)
        return eyes
        
    @property
    def eyes_irises(self):
        print('Irises detecting started')
        irises = []
        for eye in self.eyes:
            irises.extend(self.detect_irises(eye, self.skin_color))
        return irises

    @property
    def irises_color(self):
        print('Irises_color detecting started')
        irises_color = []
        for iris in self.eyes_irises:
            irises_color.append(self.detect_eyes_color(iris))
        return irises_color

    @staticmethod
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

        return faces

    @staticmethod
    def detect_eyes(image, gray, faces):
        """
        This function detects eyes.

        Arguments:

        Returns:
        eyes images names
        """

        # Load haarcascades
        eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

        # Find eyes
        eyes_images = []

        for (x, y, w, h) in faces:
            # Crop the area with faces
            roi_gray = gray[y: y + h, x: x + w]
            roi_color = image[y: y + h, x: x + w]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.2,
                minNeighbors=4,
                minSize=(150, 150)
            )

            # Save images for transfer
            for (ex, ey, ew, eh) in eyes:
                eye = roi_color[ey: ey + eh, ex: ex + ew]
                eyes_images.append(eye)

        return eyes_images

    def detect_irises(self, eye, skin_color):
        """
        This function gets eyes images, then detects irises on processed images
        and returns irises which cropped by mask.

        Arguments:

        Returns:
        """

        clone = eye.copy()

        # Get processed images
        images, images_name = self._process_image(clone)

        # Find irises and write them
        irises_images = []
        irises_images_names = []
        for image in images:
            # Find iris on current image of eye
            irises = self.find_circles_by_mask(eye.copy(), image, skin_color)

            # TODO: implement finding iris in circles
            # iris = findIris(circles)

            for j, iris in enumerate(irises):
                if iris.size != 0:
                    # Save images for transfer
                    irises_images.append(iris)
                    # Debug for manual checking: save images of eyes
                    # cv2.imwrite("./labeled/detectEye/" + name+ "_" + imagesName[i] + "_iris" + str(j) + ".jpg", iris)
                    # irisesImagesNames.append(name+ "_" + imagesName[i] + "_iris" + str(j))

        return irises_images

    @staticmethod
    def _process_image(clone):
        # Use bilateralFilter and convert to grayscale
        image = cv2.bilateralFilter(clone, 10, 100, 100)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return [grey], ['grey']

    @staticmethod
    def find_circles_by_mask(image, changed, skin_color):

        # Find circles
        rows = changed.shape[0]
        circles = cv2.HoughCircles(changed, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=100, param2=30,
                                   minRadius=10, maxRadius=100)

        crop = []
        # Find all circles with using mask
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for x, y, radius in circles[0, :]:
                center = (x, y)
                # Save original image
                clone = image.copy()

                # Create mask
                height, width, _ = clone.shape
                mask = np.zeros((height, width), np.uint8)

                # Draw on mask
                cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)

                # Copy that image using that mask
                masked_data = cv2.bitwise_and(clone, clone, mask=mask)

                # Apply Threshold
                _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for i, contour in enumerate(contours):
                    # Get the bounding rect
                    x, y, w, h = cv2.boundingRect(contour)
                    # Crop masked_data
                    iris = masked_data[y: y + h, x: x + w]
                    crop.append(iris)
                    dom_color = eyes_color.get_dominant_color(iris, 5)
                    dom_color_hsv = np.full(iris.shape, dom_color, dtype='uint8')
                    dom_color_bgr = cv2.cvtColor(dom_color_hsv, cv2.COLOR_HSV2BGR)
                    output_image = np.hstack((iris, dom_color_bgr))
                    cv2.imwrite('./labeled/detectEyeColor/' + str(x) + str(y) + '_dominantEyeColor.jpg', output_image)
        return crop

    def detect_eyes_color(self, image):
        """
        This function gets irises images, than gets dominant color on image,
        writes image with adding the area which filled by dominant color .

        Arguments:

        Returns:

        """
        output_image = image
        # TODO: add sorting irises by color and pass by wrong detected irises
        if image.size != 0:
            iris_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            dom_color = self.get_dominant_color(iris_hsv)
            dom_color_hsv = np.full(image.shape, dom_color, dtype='uint8')
            dom_color_bgr = cv2.cvtColor(dom_color_hsv, cv2.COLOR_HSV2BGR)
            output_image = np.hstack((image, dom_color_bgr))

        return output_image

    @staticmethod
    def get_dominant_color(image, k=4):
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = KMeans(n_clusters=k)
        labels = clt.fit_predict(image)
        label_counts = Counter(labels)
        dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
        return list(dominant_color)
