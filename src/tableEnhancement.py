import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

class MedicalTableEnhancer:
    def __init__(self):
        # Initialize PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        # Common medical measurement patterns
        self.measurement_patterns = [
            r'[A-Za-z]{2,}\s*[Dd]iam\s*[\d.]+\s*cm',  # Diameter measurements
            r'[A-Za-z/]{2,}\s*[\d.]+\s*cm',           # General cm measurements
            r'[A-Za-z]{2,}\s*[\d.]+\s*mm',            # mm measurements
            r'[A-Za-z]{2,}\s*[\d.]+\s*m/s',           # Velocity measurements
            r'[A-Za-z]{2,}\s*[\d.]+\s*%',             # Percentage measurements
            r'[A-Za-z]{2,}\s*[\d.]+',                 # General numeric measurements
            r'\d+\s+[A-Za-z]{2,}\s*[\d.]+',          # Numbered measurements
        ]
        
        # Keywords that indicate measurement areas
        self.measurement_keywords = [
            'diam', 'diameter', 'area', 'volume', 'velocity', 'flow',
            'cm', 'mm', 'ml', 'm/s', 'bpm', 'ao', 'la', 'rv', 'lv',
            'acs', 'ef', 'fs', 'ivs', 'pw'
        ]

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better text detection"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance contrast
        enhanced = cv2.equalizeHist(gray)
        
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return filtered

    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """Detect text regions using PaddleOCR"""
        # PaddleOCR returns results in format: [[[bbox], (text, confidence)]]
        results = self.ocr.ocr(image, cls=True)
        
        text_regions = []
        
        if results[0] is not None:
            for result in results[0]:
                # Extract bounding box coordinates
                bbox = result[0]
                text_info = result[1]
                text = text_info[0]
                confidence = text_info[1]
                
                # Convert bbox to x, y, w, h format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x = int(min(x_coords))
                y = int(min(y_coords))
                w = int(max(x_coords) - min(x_coords))
                h = int(max(y_coords) - min(y_coords))
                
                # Filter by confidence (PaddleOCR confidence is 0-1, convert to 0-100)
                conf_percent = confidence * 100
                
                if conf_percent > 30 and text.strip():  # Confidence threshold
                    text_regions.append({
                        'text': text.strip(),
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'conf': conf_percent,
                        'bbox': bbox  # Keep original bbox for advanced operations
                    })
        
        return text_regions

    def identify_measurement_regions(self, text_regions: List[Dict]) -> List[Dict]:
        """Identify regions containing medical measurements"""
        measurement_regions = []
        
        for region in text_regions:
            text = region['text'].lower()
            
            # Check if text contains measurement patterns
            is_measurement = False
            
            # Check for measurement keywords
            for keyword in self.measurement_keywords:
                if keyword in text:
                    is_measurement = True
                    break
            
            # Check for measurement patterns
            if not is_measurement:
                for pattern in self.measurement_patterns:
                    if re.search(pattern, region['text'], re.IGNORECASE):
                        is_measurement = True
                        break
            
            # Check for numeric values with units
            if not is_measurement:
                if re.search(r'\d+\.?\d*\s*(cm|mm|ml|m/s|%|bpm)', text):
                    is_measurement = True
            
            if is_measurement:
                measurement_regions.append(region)
        
        return measurement_regions

    def group_nearby_measurements(self, measurement_regions: List[Dict], 
                                threshold: int = 50) -> List[List[Dict]]:
        """Group nearby measurement regions into potential tables"""
        if not measurement_regions:
            return []
        
        groups = []
        used = set()
        
        for i, region in enumerate(measurement_regions):
            if i in used:
                continue
                
            group = [region]
            used.add(i)
            
            # Find nearby regions
            for j, other_region in enumerate(measurement_regions):
                if j in used or i == j:
                    continue
                
                # Calculate distance
                dist = np.sqrt((region['x'] - other_region['x'])**2 + 
                             (region['y'] - other_region['y'])**2)
                
                if dist < threshold:
                    group.append(other_region)
                    used.add(j)
            
            if len(group) >= 2:  # Only consider groups with multiple measurements
                groups.append(group)
        
        return groups

    def create_table_structure(self, image: np.ndarray, 
                             measurement_group: List[Dict]) -> np.ndarray:
        """Add table structure around measurement groups"""
        result_image = image.copy()
        
        if not measurement_group:
            return result_image
        
        # Find bounding box of the entire group
        min_x = min(region['x'] for region in measurement_group)
        min_y = min(region['y'] for region in measurement_group)
        max_x = max(region['x'] + region['w'] for region in measurement_group)
        max_y = max(region['y'] + region['h'] for region in measurement_group)
        
        # Add padding
        padding = 10
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(image.shape[1], max_x + padding)
        max_y = min(image.shape[0], max_y + padding)
        
        # Sort regions by position (top to bottom, left to right)
        sorted_regions = sorted(measurement_group, 
                              key=lambda r: (r['y'], r['x']))
        
        # Determine table dimensions
        rows = len(sorted_regions)
        cols = 3  # Typically: Label, Value, Unit
        
        cell_width = (max_x - min_x) // cols
        cell_height = (max_y - min_y) // rows
        
        # Draw table grid
        # Horizontal lines
        for i in range(rows + 1):
            y = min_y + i * cell_height
            cv2.line(result_image, (min_x, y), (max_x, y), (255, 255, 255), 2)
        
        # Vertical lines
        for j in range(cols + 1):
            x = min_x + j * cell_width
            cv2.line(result_image, (x, min_y), (x, max_y), (255, 255, 255), 2)
        
        # Add alternating row backgrounds
        for i in range(rows):
            if i % 2 == 1:  # Alternate rows
                y1 = min_y + i * cell_height + 1
                y2 = min_y + (i + 1) * cell_height - 1
                # Create semi-transparent overlay
                overlay = result_image.copy()
                cv2.rectangle(overlay, (min_x + 1, y1), (max_x - 1, y2), 
                            (50, 50, 50), -1)
                result_image = cv2.addWeighted(result_image, 0.8, overlay, 0.2, 0)
        
        return result_image

    def enhance_text_visibility(self, image: np.ndarray, 
                              measurement_regions: List[Dict]) -> np.ndarray:
        """Enhance text visibility in measurement regions"""
        result = image.copy()
        
        for region in measurement_regions:
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Add padding
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            # Create text background
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        return result

    def process_image(self, image: str, output_path: str = None) -> np.ndarray:
        """Main processing function"""
        # Load image
        
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        print("Processing image...")
        
        # Detect text regions (PaddleOCR works better with original colored image)
        print("Detecting text regions with PaddleOCR...")
        text_regions = self.detect_text_regions(image)
        print(f"Found {len(text_regions)} text regions")
        
        # Print detected text for debugging
        print("\nDetected text:")
        for i, region in enumerate(text_regions):
            print(f"{i+1}. '{region['text']}' (conf: {region['conf']:.1f}%)")
        
        # Identify measurement regions
        print("\nIdentifying measurement regions...")
        measurement_regions = self.identify_measurement_regions(text_regions)
        print(f"Found {len(measurement_regions)} measurement regions")
        
        # Print measurement regions for debugging
        print("\nMeasurement regions:")
        for i, region in enumerate(measurement_regions):
            print(f"{i+1}. '{region['text']}' at ({region['x']}, {region['y']})")
        
        # Group nearby measurements
        print("\nGrouping measurements...")
        measurement_groups = self.group_nearby_measurements(measurement_regions)
        print(f"Found {len(measurement_groups)} measurement groups")
        
        # Create result image
        result = image.copy()
        
        # Process each group
        for i, group in enumerate(measurement_groups):
            print(f"Processing group {i+1} with {len(group)} measurements")
            result = self.create_table_structure(result, group)
        
        # Enhance text visibility
        result = self.enhance_text_visibility(result, measurement_regions)
        
        # Save result if output path provided
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"Result saved to {output_path}")
        
        return result

    def visualize_detection(self, image_path: str):
        """Visualize the detection process step by step"""
        image = cv2.imread(image_path)
        processed = self.preprocess_image(image)
        text_regions = self.detect_text_regions(image)  # Use original image for PaddleOCR
        measurement_regions = self.identify_measurement_regions(text_regions)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Processed image
        axes[0, 1].imshow(processed, cmap='gray')
        axes[0, 1].set_title('Preprocessed Image')
        axes[0, 1].axis('off')
        
        # All text regions
        text_vis = image.copy()
        for region in text_regions:
            cv2.rectangle(text_vis, (region['x'], region['y']), 
                         (region['x'] + region['w'], region['y'] + region['h']), 
                         (0, 255, 0), 2)
            # Add text label
            cv2.putText(text_vis, f"{region['text'][:10]}...", 
                       (region['x'], region['y']-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (0, 255, 0), 1)
        axes[1, 0].imshow(cv2.cvtColor(text_vis, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'All Text Regions ({len(text_regions)})')
        axes[1, 0].axis('off')
        
        # Measurement regions
        meas_vis = image.copy()
        for region in measurement_regions:
            cv2.rectangle(meas_vis, (region['x'], region['y']), 
                         (region['x'] + region['w'], region['y'] + region['h']), 
                         (0, 0, 255), 3)
            # Add text label
            cv2.putText(meas_vis, f"{region['text'][:15]}", 
                       (region['x'], region['y']-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (0, 0, 255), 1)
        axes[1, 1].imshow(cv2.cvtColor(meas_vis, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Measurement Regions ({len(measurement_regions)})')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

# Usage example
def main(input_image):
    # Initialize with PaddleOCR
    enhancer = MedicalTableEnhancer()
    
    # Process your image
 # Replace with your image path
    output_image = 'enhanced_table_scan.jpg'
    
    try:
        # Process the image
        result = enhancer.process_image(input_image, output_image)
        
        # Visualize the detection process
        enhancer.visualize_detection(input_image)
        
        # Display result
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Enhanced Medical Scan with Table Structure (PaddleOCR)')
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error processing image: {e}")
        print("Make sure you have installed PaddleOCR: pip install paddlepaddle paddleocr")

