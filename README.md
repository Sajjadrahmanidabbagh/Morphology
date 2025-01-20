# Morphology
Use morphological operations to detect and classify basic shapes (e.g., circles, squares).

### This code is part of a course assignment (Image Processing for Engineers), Which I lectured in 2022. ###

**Code Description:**
This Python script demonstrates how to detect and classify basic geometric shapes in an image using OpenCV. It starts by loading an image and converting it to grayscale. Thresholding is applied to create a binary image, followed by morphological operations to refine the shapes. Contours are extracted, and each shape is identified based on the number of vertices in its approximated polygon. Finally, the identified shapes are annotated on the original image and saved to the disk.

**Key Libraries/Functions:**
1. OpenCV: Used for image loading (cv2.imread), preprocessing (cv2.cvtColor, cv2.threshold), morphological operations (cv2.morphologyEx), contour detection (cv2.findContours), and annotation (cv2.drawContours, cv2.putText).
2. NumPy: Utilized for array manipulations, though its role is minor here.
cv2.approxPolyDP: Key function for approximating contours to polygons, aiding in shape classification.

**Forseeable Engineering Applications:**
1. Quality Control: Used in manufacturing to ensure products meet geometric specifications by detecting and classifying shapes of components.
2. Traffic Sign Recognition: Identifying traffic signs based on their shapes (e.g., circular stop signs or triangular warning signs).
3. Robotics: Assisting robots in recognizing and manipulating objects by understanding their shapes in a structured environment.
