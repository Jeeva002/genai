import matplotlib.pyplot as plt
import cv2
import numpy as np

def enhanceTable(img):
    # Convert to grayscale if the image is colored
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # Global thresholding
    thresh, img_bin = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
    img_bin = 255 - img_bin
    plt.imshow(img_bin, cmap='gray')
    plt.title("Inverted Image with global thresholding")
    plt.show()

    # Otsu thresholding
    img_bin1 = 255 - img_gray
    thresh1, img_bin1_otsu = cv2.threshold(img_bin1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(img_bin1_otsu, cmap='gray')
    plt.title("Inverted Image with otsu thresholding")
    plt.show()

    # Fixed: Use img_gray instead of img for Otsu thresholding
    img_bin2 = 255 - img_gray
    thresh2, img_bin_otsu = cv2.threshold(img_bin2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    plt.imshow(img_bin_otsu, cmap='gray')
    plt.title("Inverted Image with otsu thresholding")
    plt.show()

    # Morphological operations for line detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    # Vertical kernel
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img_gray.shape[0]//150))
    # Use img_bin_otsu instead of undefined binary_image
    eroded_image = cv2.erode(img_bin_otsu, vertical_kernel, iterations=5)
    vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=5)

    # Horizontal kernel (define before using)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_gray.shape[1]//150, 1))
    # Use img_bin instead of undefined binary_image
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=5)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=5)

    # Combine vertical and horizontal lines
    vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)

    thresh3, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Fixed: Use grayscale image for bitwise operations
    b_image = cv2.bitwise_not(cv2.bitwise_xor(img_gray, vertical_horizontal_lines))
    plt.imshow(b_image, cmap='gray')
    plt.title("Enhanced table structure")
    plt.show()

    # Find contours
    contours, hierarchy = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fixed variable names and bounding box calculation
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda x: x[1][1]))

    # Draw bounding boxes
    boxes = []
    # Create a copy of the original image for drawing
    if len(img.shape) == 3:
        image_with_boxes = img.copy()
    else:
        image_with_boxes = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w < 1000 and h < 500):
            image_with_boxes = cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
            boxes.append([x, y, w, h])
    
    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title("Identified contours")
    plt.show()
    
    return boxes