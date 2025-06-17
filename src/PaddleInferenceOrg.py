from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np

# Initialize the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Path to your image
img_path = 'c:\\Users\\Welcome\\Downloads\\dicom2png1.png'
image = cv2.imread(img_path)

# Run OCR
result = ocr.ocr(image, cls=True)

# Extract the required information
boxes = []
txts = []
scores = []

# Process results - IMPORTANT: Handle the result structure correctly
for line in result:
    if isinstance(line, list):  # Newer versions of PaddleOCR return a list of lists
        for box in line:
            boxes.append(box[0])  # Add the box coordinates
            txts.append(box[1][0])  # Add the recognized text
            scores.append(box[1][1])  # Add the confidence score (this is a float, not a tuple)
    else:  # Older versions might return a different structure
        boxes.append(line[0])
        txts.append(line[1][0])
        scores.append(line[1][1])

# Draw results on the image
font_path = 'c:\\Users\\Welcome\\Downloads\\Humor-Sans.ttf'
print(txts)
im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)

# Save the result image
cv2.imwrite('c:\\Users\\Welcome\\Desktop\\IMG2DICOM\\dicom2png2.png', im_show)