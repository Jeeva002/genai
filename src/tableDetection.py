import cv2
import numpy as np
from paddleocr import PaddleOCR, PPStructure
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class PaddleTableDetector:
    def __init__(self, use_gpu=False, lang='en'):
        """
        Initialize PaddleOCR for table detection
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration
            lang (str): Language for OCR ('en', 'ch', etc.)
        """
        # Initialize PPStructure for layout analysis and table detection
        self.table_engine = PPStructure(
            use_gpu=use_gpu,
            show_log=True,
            lang=lang,
            layout=True,
            table=True,
            ocr=True
        )
    
    def detect_table_areas(self, image_path, save_results=True):
        """
        Detect table areas in the image
        
        Args:
            image_path (str): Path to input image
            save_results (bool): Whether to save visualization results
            
        Returns:
            dict: Detection results with table coordinates and info
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert BGR to RGB for processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use multiple contour detection methods
        edges = cv2.Canny(gray, 50, 150)
        # Perform structure analysis
        result = self.table_engine(img)
        
        # Extract table information
        table_areas = []
        for res in result:
            if res['type'] == 'table':
                bbox = res['bbox']  # [x1, y1, x2, y2]
                table_info = {
                    'bbox': bbox,
                    'confidence': res.get('score', 1.0),
                    'type': res['type']
                }
                
                # Add table structure if available
                if 'res' in res and res['res']:
                    table_info['html'] = res['res']['html']
                
                table_areas.append(table_info)
        
        # Visualize results
        if save_results and table_areas:
            self._visualize_tables(img_rgb, table_areas, image_path)
        
        return {
            'image_shape': img.shape,
            'num_tables': len(table_areas),
            'table_areas': table_areas
        }
    
    def _visualize_tables(self, img_rgb, table_areas, original_path):
        """
        Visualize detected table areas
        """
        # Create PIL image for drawing
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # Draw bounding boxes for each table
        for i, table in enumerate(table_areas):
            bbox = table['bbox']
            x1, y1, x2, y2 = bbox
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # Add label
            label = f"Table {i+1} ({table['confidence']:.2f})"
            draw.text((x1, y1-20), label, fill='red')
        
        # Display results
        plt.figure(figsize=(15, 10))
        plt.imshow(pil_img)
        plt.title(f'Detected Table Areas: {len(table_areas)}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Save visualization
        output_path = original_path.replace('.', '_table_detection.')
        pil_img.save(output_path)
        print(f"Visualization saved to: {output_path}")
    
    def extract_table_regions(self, image_path, padding=10):
        """
        Extract individual table regions as separate images
        
        Args:
            image_path (str): Path to input image
            padding (int): Padding around table area
            
        Returns:
            list: List of cropped table images
        """
        # Detect tables
        results = self.detect_table_areas(image_path, save_results=False)
        
        # Read original image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        table_images = []
        
        for i, table in enumerate(results['table_areas']):
            bbox = table['bbox']
            x1, y1, x2, y2 = bbox
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img.shape[1], x2 + padding)
            y2 = min(img.shape[0], y2 + padding)
            
            # Crop table region
            table_crop = img_rgb[int(y1):int(y2), int(x1):int(x2)]
            table_images.append({
                'image': table_crop,
                'bbox': [x1, y1, x2, y2],
                'table_id': i + 1,
                'confidence': table['confidence']
            })
            
            # Save individual table image
            table_path = image_path.replace('.', f'_table_{i+1}.')
            cv2.imwrite(table_path, cv2.cvtColor(table_crop, cv2.COLOR_RGB2BGR))
            print(f"Table {i+1} saved to: {table_path}")
        
        return table_images
    
    def get_table_coordinates(self, image_path):
        """
        Get only the coordinates of detected tables
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            list: List of table coordinates [x1, y1, x2, y2]
        """
        results = self.detect_table_areas(image_path, save_results=False)
        coordinates = []
        
        for table in results['table_areas']:
            coordinates.append(table['bbox'])
        
        return coordinates

# Usage examples
def main():
    # Initialize detector
    detector = PaddleTableDetector(use_gpu=False, lang='en')
    
    # Example image path (replace with your image)
    image_path = "c:\\Users\\Welcome\\Desktop\\IMG2DICOM\\adding_image_to_table.jpg"
    
    try:
        print("1. Detecting table areas...")
        results = detector.detect_table_areas(image_path)
        print(f"Found {results['num_tables']} table(s)")
        
        # Print table information
        for i, table in enumerate(results['table_areas']):
            bbox = table['bbox']
            print(f"Table {i+1}: x1={bbox[0]:.0f}, y1={bbox[1]:.0f}, "
                  f"x2={bbox[2]:.0f}, y2={bbox[3]:.0f}, "
                  f"confidence={table['confidence']:.3f}")
        
        print("\n2. Extracting table regions...")
        table_images = detector.extract_table_regions(image_path, padding=20)
        print(f"Extracted {len(table_images)} table images")
        
        print("\n3. Getting coordinates only...")
        coordinates = detector.get_table_coordinates(image_path)
        print("Table coordinates:", coordinates)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install PaddleOCR: pip install paddlepaddle paddleocr")
        print("And replace 'your_table_image.jpg' with your actual image path")

if __name__ == "__main__":
    main()