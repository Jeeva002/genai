import torch
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from PIL import Image, ImageDraw
import requests
import numpy as np

class TableDetector:
    def __init__(self):
        """Initialize the Table Transformer model for table detection."""
        # Load the pre-trained model and processor
        self.model_name = "microsoft/table-transformer-detection"
        self.processor = DetrImageProcessor.from_pretrained(self.model_name)
        self.model = TableTransformerForObjectDetection.from_pretrained(self.model_name)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def detect_tables(self, image_path, confidence_threshold=0.7):
        """
        Detect tables in an image.
        
        Args:
            image_path (str): Path to the image file or URL
            confidence_threshold (float): Minimum confidence score for detections
            
        Returns:
            dict: Dictionary containing detected tables and their coordinates
        """
        # Load image
        if image_path.startswith('http'):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=confidence_threshold
        )[0]
        
        # Extract table information
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            tables.append({
                "confidence": round(score.item(), 3),
                "label": self.model.config.id2label[label.item()],
                "bbox": box,  # [x_min, y_min, x_max, y_max]
                "area": (box[2] - box[0]) * (box[3] - box[1])
            })
        
        return {
            "image_size": image.size,
            "num_tables": len(tables),
            "tables": tables,
            "original_image": image
        }
    
    def visualize_detections(self, detection_results, save_path=None):
        """
        Visualize detected tables by drawing bounding boxes.
        
        Args:
            detection_results (dict): Results from detect_tables method
            save_path (str, optional): Path to save the annotated image
            
        Returns:
            PIL.Image: Annotated image with bounding boxes
        """
        image = detection_results["original_image"].copy()
        draw = ImageDraw.Draw(image)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, table in enumerate(detection_results["tables"]):
            bbox = table["bbox"]
            confidence = table["confidence"]
            label = table["label"]
            
            # Choose color
            color = colors[i % len(colors)]
            
            # Draw bounding box
            draw.rectangle(bbox, outline=color, width=3)
            
            # Add label and confidence
            text = f"{label}: {confidence:.2f}"
            draw.text((bbox[0], bbox[1] - 20), text, fill=color)
        
        if save_path:
            image.save(save_path)
            
        return image

# Example usage
def main():
    # Initialize detector
    detector = TableDetector()
    
    # Example 1: Detect tables from a local image
    print("Example 1: Local image detection")
    try:
        # Replace with your image path
        results = detector.detect_tables("c:\\Users\\Welcome\\Desktop\\IMG2DICOM\\adding_image_to_table.jpg", confidence_threshold=0.7)
        
        print(f"Image size: {results['image_size']}")
        print(f"Number of tables detected: {results['num_tables']}")
        
        for i, table in enumerate(results['tables']):
            print(f"\nTable {i+1}:")
            print(f"  Confidence: {table['confidence']}")
            print(f"  Label: {table['label']}")
            print(f"  Bounding box: {table['bbox']}")
            print(f"  Area: {table['area']:.2f}")
        
        # Visualize results
        annotated_image = detector.visualize_detections(results, "detected_tables.jpg")
        print("Annotated image saved as 'detected_tables.jpg'")
        
    except Exception as e:
        print(f"Error with local image: {e}")
    
    # Example 2: Detect tables from a URL
    print("\nExample 2: URL image detection")
    try:
        # Example image URL with tables
        url = "https://example.com/image_with_tables.jpg"
        results = detector.detect_tables(url, confidence_threshold=0.6)
        
        print(f"Number of tables detected: {results['num_tables']}")
        
        # Sort tables by confidence
        sorted_tables = sorted(results['tables'], key=lambda x: x['confidence'], reverse=True)
        
        for i, table in enumerate(sorted_tables):
            print(f"Table {i+1}: {table['label']} (confidence: {table['confidence']:.3f})")
            
    except Exception as e:
        print(f"Error with URL image: {e}")

# Additional utility functions
def batch_detect_tables(image_paths, output_dir="output"):
    """
    Process multiple images for table detection.
    
    Args:
        image_paths (list): List of image paths
        output_dir (str): Directory to save results
    """
    import os
    
    detector = TableDetector()
    os.makedirs(output_dir, exist_ok=True)
    
    results_summary = []
    
    for i, image_path in enumerate(image_paths):
        try:
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            results = detector.detect_tables(image_path)
            
            # Save annotated image
            output_path = os.path.join(output_dir, f"annotated_{i+1}.jpg")
            detector.visualize_detections(results, output_path)
            
            # Store summary
            results_summary.append({
                "image_path": image_path,
                "num_tables": results["num_tables"],
                "tables": results["tables"]
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results_summary.append({
                "image_path": image_path,
                "error": str(e)
            })
    
    return results_summary

def filter_tables_by_size(detection_results, min_area=1000):
    """
    Filter detected tables by minimum area.
    
    Args:
        detection_results (dict): Results from detect_tables
        min_area (float): Minimum area threshold
        
    Returns:
        dict: Filtered results
    """
    filtered_tables = [
        table for table in detection_results["tables"]
        if table["area"] >= min_area
    ]
    
    detection_results["tables"] = filtered_tables
    detection_results["num_tables"] = len(filtered_tables)
    
    return detection_results

if __name__ == "__main__":
    main()

# Installation requirements:
"""
pip install torch torchvision
pip install transformers
pip install Pillow
pip install requests
"""