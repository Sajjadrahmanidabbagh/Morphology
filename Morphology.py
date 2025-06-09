
"""
Created on Tue Jan 21 15:45:57 2022

@author: sajjad
"""

import cv2
import numpy as np

def load_and_preprocess_image(image_path):
    """
    Load the input image and preprocess it by converting to grayscale and thresholding.
    Args:
        image_path (str): Path to the input image.
    Returns:
        tuple: Grayscale image and binary thresholded image.
    """
    # Load image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    return image, binary

def apply_morphological_operations(binary_image):
    """
    Apply morphological operations to refine shapes in the binary image.
    Args:
        binary_image (np.ndarray): Binary thresholded image.
    Returns:
        np.ndarray: Refined binary image after morphological operations.
    """
    # Define kernel for morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Apply closing to fill small holes
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Apply opening to remove noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return opened

def detect_shapes(refined_image, original_image):
    """
    Detect and classify shapes in the image using contours and polygon approximation.
    Args:
    
        refined_image (np.ndarray): Image after morphological operations.
        original_image (np.ndarray): Original input image for annotation.
        
    Returns:
        np.ndarray: Annotated image with detected shapes.
    """
    # Find contours
    contours, _ = cv2.findContours(refined_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Determine the shape based on the number of vertices
        x, y, w, h = cv2.boundingRect(approx)
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            aspect_ratio = float(w) / h
            shape = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
        elif len(approx) > 5:
            shape = "Circle"
        else:
            shape = "Polygon"

        # Annotate the image
        cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 2)
        cv2.putText(original_image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return original_image

def main(image_path, output_path):
    """
    Main function to perform shape detection and save the result.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the annotated image.
    Returns:
        None
    """
    # Load and preprocess image
    original_image, binary_image = load_and_preprocess_image(image_path)

    # Apply morphological operations
    refined_image = apply_morphological_operations(binary_image)

    # Detect shapes and annotate the image
    annotated_image = detect_shapes(refined_image, original_image)

    # Save and display the result
    cv2.imwrite(output_path, annotated_image)
    cv2.imshow("Detected Shapes", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_image_path = "shapes.jpg"  # Replace with your input image path
    output_image_path = "annotated_shapes.jpg"  # Replace with your desired output path
    main(input_image_path, output_image_path)
