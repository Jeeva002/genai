import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
from OcrPreprocessing import main
from paddleInference import extractTextFromImage
class UltrasoundTableDetector:
    def __init__(self):
        # Initialize PaddleOCR (supports multiple languages)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_thresh=0.3,      # Lower threshold for text detection
        det_db_box_thresh=0.5,  # Lower box threshold
        det_db_unclip_ratio=2.0)
    def create_roi_on_white_background(self,image, roi_coords):
            """
            Create a white image with only the ROI region filled
            White background is usually better for OCR
            """
            # Get original image dimensions
            height, width = image.shape[:2]
            
            # Create white background
            if len(image.shape) == 3:
                white_image = np.full_like(image, 255)  # RGB white
            else:
                white_image = np.full_like(image, 255)  # Grayscale white
            
            # Extract ROI coordinates
            y, h, x, w = roi_coords[1],roi_coords[3],roi_coords[0],roi_coords[2]
            
            # Copy ROI from original to white image at same position
            white_image[y:y+h, x:x+w] = image[y:y+h, x:x+w]
            
            return white_image
    def detect_measurement_table(self, image):
        """
        Main function to detect measurement tables in ultrasound images
        """
        # 1. Find all rectangular regions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use multiple contour detection methods
        edges = cv2.Canny(gray, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow('contour',edges)
        # cv2.waitKey(0)
 
    
        candidate_regions = []
        
        for contour in contours:
            
            
            # Filter by area and aspect ratio
            area = cv2.contourArea(contour)

           # print("contour area",area)
            if area < 1: # Too small
                continue

            # Check if roughly rectangular
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # print("area",area,epsilon,approx)
            # cv2.drawContours(image, contour,-1, (0, 255, 0), 3)
            # cv2.imshow('contour',image)
            # cv2.waitKey(0)  
            if len(approx) >= 4:  # Roughly rectangular
                # cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
               # print(area)

                x, y, w, h = cv2.boundingRect(contour)

                # Extract region and analyze text density
                roi = image[y:y+h, x:x+w]  # Use color image for PaddleOCR
                # print("len",len(approx),x,y,w,h)
                # cv2.imshow('roi',roi)
                # cv2.waitKey(0)
                
                text_density = self.calculate_text_density_paddleocr(roi)
                line_structure = self.analyze_line_structure(gray[y:y+h, x:x+w])
                measurement_score = self.calculate_measurement_score(roi)

                candidate_regions.append({
                    'bbox': (x, y, w, h),
                    'text_density': text_density,
                    'line_structure': line_structure,
                    'measurement_score': measurement_score,
                    'area': area,
                    'aspect_ratio': w / h
                })
        
        # Score and rank candidates
        return self.rank_table_candidates(candidate_regions)
    
    def calculate_text_density_paddleocr(self, roi):
        """
        Calculate text density using PaddleOCR
        """
        try:
            # Run PaddleOCR on the region
            result = self.ocr.ocr(roi, cls=True)
            
            if not result or not result[0]:
                return 0
            
            # Count words and characters
            total_text = ""
            word_count = 0
            confidence_sum = 0
            detection_count = 0
            
            for line in result[0]:
                if line:
                    text = line[1][0]  # Extract text
                    confidence = line[1][1]  # Extract confidence
                    
                    total_text += text + " "
                    word_count += len(text.split())
                    confidence_sum += confidence
                    detection_count += 1
            
            # Calculate density metrics
            roi_area = roi.shape[0] * roi.shape[1]
            text_density = word_count / roi_area * 10000  # Normalize
            
            # Weight by average confidence
            avg_confidence = confidence_sum / detection_count if detection_count > 0 else 0
            weighted_density = text_density * avg_confidence
            
            return weighted_density
            
        except Exception as e:
            print(f"Error in PaddleOCR text density calculation: {e}")
            return 0
    
    def calculate_measurement_score(self, roi):
        """
        Calculate how likely this region contains measurement data
        """
        try:
            result = self.ocr.ocr(roi, cls=True)
            
            if not result or not result[0]:
                return 0
            
            # Extract all text
            all_text = ""
            for line in result[0]:
                if line:
                    text = line[1][0]
                    all_text += text.lower() + " "
            
            # Measurement indicators (flexible patterns)
            measurement_patterns = [
                r'\d+\.?\d*\s*(?:mm|cm|ml|sec|bpm|%|hz)',  # Numbers with units
                r'[a-zA-Z]+\s*[:=]\s*\d+',                # Key-value pairs
                r'\b(?:diameter|length|area|volume|velocity|depth|width|height)\b',  # Common terms
                r'\d+\.\d+',                               # Decimal numbers
                r'\d+\s*x\s*\d+',                         # Dimensions (e.g., "12 x 34")
                r'(?:left|right|anterior|posterior|superior|inferior)',  # Anatomical terms
                r'(?:systolic|diastolic|mean|peak|max|min)', # Medical measurement terms
            ]
            
            score = 0
            for pattern in measurement_patterns:
                matches = re.findall(pattern, all_text)
                score += len(matches)
            
            # Bonus for colon patterns (key:value)
            colon_patterns = all_text.count(':')
            score += colon_patterns * 2
            
            # Bonus for equal sign patterns
            equal_patterns = all_text.count('=')
            score += equal_patterns * 2
            
            return score
            
        except Exception as e:
            print(f"Error in measurement score calculation: {e}")
            return 0
    
    def analyze_line_structure(self, roi_gray):
        """
        Analyze line structure to identify table-like patterns
        """
        # Look for horizontal line patterns (table rows)
        if roi_gray.shape[1] == 0 or roi_gray.shape[0] == 0:
            return 0
            
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, roi_gray.shape[1]//10), 1))
        horizontal_lines = cv2.morphologyEx(roi_gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Count distinct horizontal structures
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Also check for text line patterns using PaddleOCR
        try:
            roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
            result = self.ocr.ocr(roi_color, cls=True)
            
            if result and result[0]:
                # Group text detections by similar Y coordinates (lines)
                y_coords = []
                for line in result[0]:
                    if line:
                        bbox = line[0]  # Bounding box coordinates
                        # Calculate center Y coordinate
                        center_y = (bbox[0][1] + bbox[2][1]) / 2
                        y_coords.append(center_y)
                
                # Count distinct lines (cluster Y coordinates)
                if y_coords:
                    y_coords.sort()
                    distinct_lines = 1
                    for i in range(1, len(y_coords)):
                        if abs(y_coords[i] - y_coords[i-1]) > 10:  # Threshold for line separation
                            distinct_lines += 1
                    
                    return max(len(contours), distinct_lines)
            
        except Exception as e:
            print(f"Error in line structure analysis: {e}")
        
        return len(contours)
    
    def rank_table_candidates(self, candidate_regions):
        """
        Rank candidates based on multiple criteria
        """
        if not candidate_regions:
            return []
        
        # Calculate composite scores
        for candidate in candidate_regions:
            # Normalize scores
            text_score = min(candidate['text_density'], 100) / 100  # Cap at 100
            structure_score = min(candidate['line_structure'], 10) / 10  # Cap at 10
            measurement_score = min(candidate['measurement_score'], 20) / 20  # Cap at 20
            
            # Size preference (medium to large regions)
            area_score = min(candidate['area'], 50000) / 50000
            
            # Aspect ratio preference (wider is better for tables)
            aspect_score = min(candidate['aspect_ratio'], 3) / 3
            
            # Composite score with weights
            composite_score = (
                text_score * 0.3 +
                structure_score * 0.2 +
                measurement_score * 0.3 +
                area_score * 0.1 +
                aspect_score * 0.1
            )
            
            candidate['composite_score'] = composite_score
        
        # Sort by composite score (highest first)
        ranked_candidates = sorted(candidate_regions, key=lambda x: x['composite_score'], reverse=True)
        
        # Filter out very low scoring candidates
        filtered_candidates = [c for c in ranked_candidates if c['composite_score'] > 0.1]
        
        return filtered_candidates
    
    def extract_table_content(self, image, bbox):
        """
        Extract and structure the content from detected table region
        """
        x, y, w, h = bbox
        roiValues=[x,y,w,h]
        roi = image[y:y+h, x:x+w]
        print("ratio",y,y,h,x,x,w)
        cv2.imwrite('c:\\Users\\Welcome\\Desktop\\IMG2DICOM\\new.jpg',roi)
        cv2.imshow('image55',roi)
        cv2.waitKey(0)
        try:
            print("shape",roi.shape)
            height, width = roi.shape[:2]
            upscaled = cv2.resize(roi, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
            
            whiteROI=self.create_roi_on_white_background(image,roiValues)
            original=extractTextFromImage(upscaled)
            lab = cv2.cvtColor(whiteROI, cv2.COLOR_BGR2LAB)

            # Apply CLAHE only to the L (lightness) channel
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])

            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            main(roi)
            cv2.imshow('white',whiteROI)                             
            cv2.imshow('enhanced',enhanced)
            cv2.imshow('roi',roi)
            cv2.imshow('upscale',upscaled)
            cv2.waitKey(0)
            result = self.ocr.ocr(upscaled, cls=True)

            # print("original",original)
            print("Result",result)
            if not result or not result[0]:
                return {}
            
            # Extract text with positions
            text_elements = []
            for line in result[0]:
                print("line",line)
                if line:
                    bbox_coords = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # Calculate center position
                    center_x = (bbox_coords[0][0] + bbox_coords[2][0]) / 2
                    center_y = (bbox_coords[0][1] + bbox_coords[2][1]) / 2
                    
                    text_elements.append({
                        'text': text,
                        'x': center_x,
                        'y': center_y,
                        'confidence': confidence
                    })
                    print("text elementtt",text_elements)
            
            # Group into potential key-value pairs
            structured_data = self.structure_table_data(text_elements)

            return structured_data
            
        except Exception as e:
            print(f"Error extracting table content: {e}")
            return {}
    
    def structure_table_data(self, text_elements):
        """
        Attempt to structure text elements into key-value pairs
        """
        structured = {}
        
        # Sort by Y coordinate (top to bottom)
        text_elements.sort(key=lambda x: x['y'])

        # Group elements by similar Y coordinates (same line)
        lines = []
        current_line = []
        
        for element in text_elements:
            if not current_line:
                current_line.append(element)
            else:
                # Check if element is on the same line
                if abs(element['y'] - current_line[-1]['y']) < 15:  # Same line threshold
                    current_line.append(element)
                else:
                    # Sort current line by X coordinate (left to right)
                    current_line.sort(key=lambda x: x['x'])
                    lines.append(current_line)
                    current_line = [element]
        
        # Add the last line
        if current_line:
            current_line.sort(key=lambda x: x['x'])
            lines.append(current_line)
        
        # Try to extract key-value pairs from each line
        for line in lines:
                if len(line) >= 2:
                    # Assume first element is key, second is value
                    key = line[0]['text'].strip()
                    value = line[1]['text'].strip()
                    
                    # Clean up key (remove colons, equals signs)
                    key = re.sub(r'[:=]\s*$', '', key)
                    
                    structured[key] = value
                    
                elif len(line) == 1:
                    # Handle single element lines
                    text = line[0]['text'].strip()
                    
                    # Option 1: Check if text contains delimiters like : or =
                    if ':' in text or '=' in text:
                        # Split by delimiter
                        if ':' in text:
                            parts = text.split(':', 1)  # Split only on first occurrence
                        elif '=' in text:
                            parts = text.split('=', 1)
                        
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            structured[key] = value
                            continue
                    
                    # Option 2: Check if text ends with 'cm' or number + space + 'cm'
                    # Pattern: text ending with digits followed by optional space and 'cm'
                    cm_pattern = r'^(.+?)\s*(\d+(?:\.\d+)?\s*cm)$'
                    match = re.match(cm_pattern, text, re.IGNORECASE)
                    
                    if match:
                        key = match.group(1).strip()
                        value = match.group(2).strip()
                        
                        # Clean up key (remove any trailing punctuation)
                        key = re.sub(r'[^\w\s]+$', '', key).strip()
                        
                        structured[key] = value
                    else:
                        # If no pattern matches, check for other common patterns
                        # Pattern: key followed by space and value (like "1 D 0.22cm")
                        space_pattern = r'^([^\d]+)\s+(.+)$'
                        space_match = re.match(space_pattern, text)
                        
                        if space_match:
                            key = space_match.group(1).strip()
                            value = space_match.group(2).strip()
                            structured[key] = value
                        else:
                            # If no pattern found, use text as key with empty value
                            structured[text] = ""
            
        print("struct", structured)
        return structured


# Usage example
def detect_tables_in_ultrasound(image):
    """
    Example usage function
    """
    # Load image
    # image = cv2.imread(image_path)
    # if image is None:
    #     print(f"Could not load image: {image_path}")
    #     return
    
    # Initialize detector
    detector = UltrasoundTableDetector()
    
    # Detect tables
    candidates = detector.detect_measurement_table(image)
    
    print(f"Found {len(candidates)} table candidates:")
    
    for i, candidate in enumerate(candidates[:1]):
        print(f"\nCandidate {i+1}:")
        print(f"  Bbox: {candidate['bbox']}")
        print(f"  Composite Score: {candidate['composite_score']:.3f}")
        print(f"  Text Density: {candidate['text_density']:.3f}")
        print(f"  Measurement Score: {candidate['measurement_score']}")
        print(f"  Line Structure: {candidate['line_structure']}")
        
        # Extract content from top candidate
        if i == 0:  # Only for the best candidate
            print("inside i")
            structured_data = detector.extract_table_content(image, candidate['bbox'])
            if structured_data:
                print("  Extracted Data:")
                for key, value in structured_data.items():
                    print(f"    {key}: {value}")
    
    # Visualize results
    result_image = image.copy()
    for i, candidate in enumerate(candidates[:3]):  # Show top 3 candidates
        x, y, w, h = candidate['bbox']
        color = [(0, 255, 0), (0, 255, 255), (255, 0, 255)][i]  # Different colors
        cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(result_image, f"Table {i+1}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow('Detected Tables', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return candidates, result_image

# Example usage:
# candidates, result_img = detect_tables_in_ultrasound('ultrasound_image.jpg')
# cv2.imshow('Detected Tables', result_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()