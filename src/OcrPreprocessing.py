import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import os

def preprocess_image_for_ocr(image):
    """
    Apply multiple preprocessing techniques to improve OCR accuracy for small text
    """
    # Add input validation
    if image is None:
        raise ValueError("Input image is None. Check if the image file exists and can be loaded.")
    
    if image.size == 0:
        raise ValueError("Input image is empty.")
    
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Method 1: Simple upscaling with interpolation
    def method1_upscale(img):
        # Scale up by 3x using cubic interpolation
        height, width = img.shape[:2]
        upscaled = cv2.resize(img, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        return upscaled
    
    # Method 2: Adaptive thresholding + upscaling
    def method2_adaptive_threshold(img):
        # Apply adaptive thresholding to improve contrast
        adaptive_thresh = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        # Scale up
        height, width = adaptive_thresh.shape[:2]
        upscaled = cv2.resize(adaptive_thresh, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        return upscaled
    
    # Method 3: Morphological operations + upscaling
    def method3_morphological(img):
        # Apply morphological closing to connect broken characters
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # Apply Gaussian blur to smooth edges
        blurred = cv2.GaussianBlur(morph, (3, 3), 0)
        
        # Scale up
        height, width = blurred.shape[:2]
        upscaled = cv2.resize(blurred, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        return upscaled
    
    # Method 4: Sharpening + contrast enhancement
    def method4_sharpen_contrast(img):
        # Apply sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(sharpened)
        
        # Scale up
        height, width = contrast_enhanced.shape[:2]
        upscaled = cv2.resize(contrast_enhanced, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        return upscaled
    
    # Method 5: Bilateral filter + edge preservation
    def method5_bilateral_filter(img):
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Apply unsharp masking
        gaussian = cv2.GaussianBlur(filtered, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(filtered, 1.5, gaussian, -0.5, 0)
        
        # Scale up
        height, width = unsharp_mask.shape[:2]
        upscaled = cv2.resize(unsharp_mask, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        return upscaled
    
    # Try all methods and return them
    methods = {
        'original': gray,
        'method1_upscale': method1_upscale(gray),
        'method2_adaptive': method2_adaptive_threshold(gray),
        'method3_morphological': method3_morphological(gray),
        'method4_sharpen': method4_sharpen_contrast(gray),
        'method5_bilateral': method5_bilateral_filter(gray)
    }
    
    return methods

def extractTextFromImage(image):
    """
    Enhanced OCR extraction with preprocessing
    """
    # Add input validation
    if image is None:
        print("Error: Image is None")
        return [], [], "none"
    
    if image.size == 0:
        print("Error: Image is empty")
        return [], [], "none"
    
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    
    # Initialize OCR with optimized parameters for small text
    ocr = PaddleOCR(
        use_angle_cls=True, 
        lang='en',
        det_db_thresh=0.3,      # Lower threshold for text detection
        det_db_box_thresh=0.5,  # Lower box threshold
        det_db_unclip_ratio=2.0 # Higher unclip ratio for small text
    )
    
    # Preprocess the image
    try:
        processed_images = preprocess_image_for_ocr(image)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return [], [], "none"
    
    best_result = None
    best_score = 0
    best_method = ""
    best_image = None
    
    # Try OCR on each preprocessed version
    for method_name, processed_img in processed_images.items():
        print(f"Testing method: {method_name}")
        
        try:
            # Run OCR
            result = ocr.ocr(processed_img, cls=True)
            
            # Handle empty results
            if not result or not result[0]:
                print(f"  No text detected with {method_name}")
                continue
            
            # Calculate average confidence score
            total_score = 0
            text_count = 0
            detected_texts = []
            
            for line in result:
                if isinstance(line, list):
                    for box in line:
                        if box and len(box) >= 2:  # Ensure box has proper structure
                            detected_texts.append(box[1][0])
                            total_score += box[1][1]
                            text_count += 1
                else:
                    if line and len(line) >= 2:  # Ensure line has proper structure
                        detected_texts.append(line[1][0])
                        total_score += line[1][1]
                        text_count += 1
            
            avg_score = total_score / text_count if text_count > 0 else 0
            
            print(f"  Detected texts: {detected_texts}")
            print(f"  Average confidence: {avg_score:.3f}")
            print(f"  Text count: {text_count}")
            
            # Check if this method detected more text or has better confidence
            if text_count > best_score or (text_count == best_score and avg_score > best_score):
                best_result = result
                best_score = max(text_count, avg_score)
                best_method = method_name
                best_image = processed_img
                
        except Exception as e:
            print(f"  Error with {method_name}: {e}")
            continue
    
    print(f"\nBest method: {best_method}")
    
    # Process the best result
    if best_result and best_image is not None:
        boxes = []
        txts = []
        scores = []
        
        for line in best_result:
            if isinstance(line, list):
                for box in line:
                    if box and len(box) >= 2:
                        boxes.append(box[0])
                        txts.append(box[1][0])
                        scores.append(box[1][1])
            else:
                if line and len(line) >= 2:
                    boxes.append(line[0])
                    txts.append(line[1][0])
                    scores.append(line[1][1])
        
        print(f"Final detected texts: {txts}")
        
        # Optional: Draw results
        try:
            font_path = 'c:\\Users\\Welcome\\Downloads\\Humor-Sans.ttf'
            if os.path.exists(font_path):
                im_show = draw_ocr(best_image, boxes, txts, scores, font_path=font_path)
            else:
                print("Font file not found, using default font")
                im_show = draw_ocr(best_image, boxes, txts, scores)
            
            # Save the best preprocessed image and result
            cv2.imwrite('best_preprocessed.jpg', best_image)
            cv2.imwrite('ocr_result.jpg', im_show)
            
        except Exception as e:
            print(f"Error drawing results: {e}")
        
        return txts, scores, best_method
    
    return [], [], "none"

def preprocess_for_small_numbers(image):
    """
    Specific preprocessing for detecting small numbers like '2' in 'D2'
    """
    if image is None:
        raise ValueError("Input image is None")
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Step 1: Upscale significantly
    height, width = gray.shape[:2]
    upscaled = cv2.resize(gray, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
    
    # Step 2: Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(upscaled)
    
    # Step 3: Apply slight Gaussian blur to smooth pixelation
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Step 4: Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return binary

# Main execution with proper error handling
def main(image):
    img_path = 'c:\\Users\\Welcome\\Desktop\\IMG2DICOM\\new.jpg'
    
    # # Check if file exists
    # if not os.path.exists(img_path):
    #     print(f"Error: Image file not found at {img_path}")
    #     print("Please check the file path and ensure the image exists.")
    #     return
    
    # # Load image with error handling
    # image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not load image from {img_path}")
        print("This could be due to:")
        print("1. File doesn't exist")
        print("2. File is corrupted")
        print("3. Unsupported image format")
        print("4. Insufficient permissions")
        return
    
    print(f"Successfully loaded image with shape: {image.shape}")
    
    # Crop the image
    cropped_image = image
    
    # Check if cropped region is valid
    if cropped_image.size == 0:
        print("Error: Cropped region is empty. Check your cropping coordinates.")
        print(f"Original image shape: {image.shape}")
        print("Crop coordinates: [669:767, 1002:1136]")
        return
    
    print(f"Cropped image shape: {cropped_image.shape}")
    
    # Save cropped image for debugging
    cv2.imwrite('cropped_debug.jpg', cropped_image)
    print("Saved cropped image as 'cropped_debug.jpg' for inspection")
    
    # Extract text
    texts, scores, best_method = extractTextFromImage(cropped_image)
    
    if texts:
        print(f"\nExtracted texts using {best_method}:")
        for i, (text, score) in enumerate(zip(texts, scores)):
            print(f"  {i+1}. '{text}' (confidence: {score:.3f})")
    else:
        print("No text was detected with any method.")
    
    # Try the specialized preprocessing as well
    print("\nTrying specialized preprocessing for small numbers...")
    try:
        processed_img = preprocess_for_small_numbers(cropped_image)
        cv2.imwrite('processed_small_numbers.jpg', processed_img)
        print("Saved specialized preprocessing result as 'processed_small_numbers.jpg'")
        
        # Initialize OCR for specialized processing
        ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_thresh=0.3)
        result = ocr.ocr(processed_img, cls=True)
        
        if result and result[0]:
            print("Results from specialized preprocessing:")
            for line in result:
                if isinstance(line, list):
                    for box in line:
                        if box and len(box) >= 2:
                            print(f"  '{box[1][0]}' (confidence: {box[1][1]:.3f})")
        else:
            print("No text detected with specialized preprocessing")
            
    except Exception as e:
        print(f"Error in specialized preprocessing: {e}")

if __name__ == "__main__":
    main()