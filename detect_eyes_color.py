# import the necessary packages
import os
import cv2
from lib.detectEyesLibClass import Face
from PIL import Image
import scipy.misc

if __name__ == '__main__':

    @classmethod
    # TODO: implement this method
    def make_dir(draft_id):
        file_dir = os.path.join(draft_id)
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        return file_dir

    # Load images of faces
    facesImages = ('face1', 'face2', 'face3')

    for j, name in enumerate(facesImages):
        irises_colors = Face(cv2.imread('./pictures/faces/' + name + '.jpg')).irises_color
        for iris in irises_colors:
            # im = Image.fromarray(iris)
            # im.save("./labeled/" + name + "_iris" + str(j) + ".jpg")
            cv2.imwrite('./labeled/' + name + '_dominantEyeColor_' + str(j) + '.jpg', iris)
    print('Debug: The end of detecting eyes')
