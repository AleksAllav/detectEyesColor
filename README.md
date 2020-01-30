# Eye color detection

This program detects the color of a person's eyes from a photo of their face. The program executes it by this main steps:

1. The searching faces,

2. The searching eyes on the face,

3. The searching the iris on the eye,

4. The detecting dominant color on the iris.


## More about each step

1. The searching faces:

Face search is performed by the method face_cascade.detectMultiScale(), where face_cascade is pre-trained classifiers for face from 'opencv/data/haarcascades/' folder. If face isn't detected by that method, the function returns whole image.
This step is for narrowing area for eye search. 

2. The searching eyes on face:

Eyes search is same with searching faces.

3. The searching the iris on the eye:

Iris search uses another method. Firstly, the function converts eye's image to grayscale with using bilateralFilter (this method showed best results on tests). Secondly, circles are searched by method cv2.HoughCircles(). Tham function creates mask, finds the minimal bounding rect for every circle and returns croped masked data.

4. The detecting dominant color on the iris:

The detecting dominant color is performed by using the sklearn.cluster KMeans package.


## Examples current perfoming

Good examples:

<img src="https://github.com/AleksAllav/detectEyesColor/blob/master/pictures/eyeColors/face1_eye2_grey_iris0_dominantEyeColor.jpg" width="48">

<img src="https://github.com/AleksAllav/detectEyesColor/blob/master/pictures/eyeColors/face2_eye2_grey_iris0_dominantEyeColor.jpg" width="48">

The bad example:

<img src="https://github.com/AleksAllav/detectEyesColor/blob/master/pictures/eyeColors/face3_eye2_grey_iris0_dominantEyeColor.jpg" width="48">


