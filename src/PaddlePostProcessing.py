import pydicom
import cv2
import numpy as np
import sys
import os
from paddleInference import extractTextFromImage
# Global variable for the image (needed for mouse callback)
img = None

def dicom_to_png(dicom_path, output_path=None):
    """
    Convert a DICOM file to PNG format using OpenCV
    
    Args:
        dicom_path (str): Path to the input DICOM file
        output_path (str): Path for the output PNG file (optional)
    
    Returns:
        tuple: (success_bool, pixel_array)
    """
    try:
        # Read the DICOM file
        dicom_data = pydicom.dcmread(dicom_path)
        print("DICOM file loaded successfully")
        
        # Extract pixel array
        pixel_array = dicom_data.pixel_array
        print(f"Original pixel array shape: {pixel_array.shape}")
        print(f"Original pixel array dtype: {pixel_array.dtype}")
        
        # Handle different bit depths and normalize to 8-bit
        if pixel_array.dtype != np.uint8:
            # Normalize to 0-255 range
            pixel_min = pixel_array.min()
            pixel_max = pixel_array.max()
            
            if pixel_max > pixel_min:
                # Normalize to 0-255
                pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
            else:
                # Handle case where all pixels have the same value
                pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        
        # Handle different photometric interpretations
        display_array = pixel_array.copy()
        
        if hasattr(dicom_data, 'PhotometricInterpretation'):
            photometric = dicom_data.PhotometricInterpretation
            print(f"Photometric Interpretation: {photometric}")
            
            if photometric == 'MONOCHROME1':
                # Invert the image for MONOCHROME1 (lower values = brighter)
                display_array = 255 - pixel_array
            elif photometric == 'RGB':
                # For ultrasound images, try different approaches
                print("Detected RGB ultrasound image")
                
                # Option 1: Keep original RGB (for display)
                display_rgb = pixel_array.copy()
                
                # Option 2: Convert RGB to BGR for OpenCV display
                display_bgr = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2BGR)
                
                # Option 3: Try without color conversion (sometimes ultrasound data is already in correct format)
                display_direct = pixel_array.copy()
                
                # Display all three versions to compare
                cv2.imshow('Original RGB', display_rgb)
                cv2.imshow('RGB to BGR', display_bgr) 
                cv2.imshow('Direct (no conversion)', display_direct)
                
                print("Three versions displayed:")
                print("1. Original RGB")
                print("2. RGB to BGR converted")
                print("3. Direct (no conversion)")
                print("Compare and choose the best looking one")
                
                # Use the direct version as default (often works best for ultrasound)
                display_array = display_direct
            else:
                print(f"Unknown photometric interpretation: {photometric}")
        else:
            print("No photometric interpretation found, using direct pixel data")
        
        # Show single result if not RGB
        if not (hasattr(dicom_data, 'PhotometricInterpretation') and 
                dicom_data.PhotometricInterpretation == 'RGB'):
            cv2.imshow('DICOM Image', display_array)
        
        print("Image displayed. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Create output filename if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(dicom_path))[0]
            output_path = f"{base_name}.png"
        
        # For saving, use the pixel_array without BGR conversion for RGB images
        save_array = pixel_array
        if (hasattr(dicom_data, 'PhotometricInterpretation') and 
            dicom_data.PhotometricInterpretation == 'RGB'):
            # OpenCV expects BGR for saving, so convert RGB to BGR
            save_array = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2BGR)
        
        # Save as PNG using OpenCV
        success = cv2.imwrite(output_path, save_array)
        
        if success:
            print(f"Successfully converted '{dicom_path}' to '{output_path}'")
            print(f"Image dimensions: {pixel_array.shape}")
            return True, pixel_array
        else:
            print(f"Failed to save PNG file: {output_path}")
            return False, pixel_array
            
    except FileNotFoundError:
        print(f"Error: DICOM file '{dicom_path}' not found")
        return False, None
    except Exception as e:
        print(f"Error converting DICOM file: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False, None

# Alternative function that gives you more control over color handling
def dicom_to_png_advanced(dicom_path, output_path=None, color_mode='auto'):
    """
    Advanced DICOM to PNG converter with color mode options
    
    Args:
        dicom_path (str): Path to the input DICOM file
        output_path (str): Path for the output PNG file (optional)
        color_mode (str): 'auto', 'rgb', 'bgr', 'gray', 'no_convert'
    
    Returns:
        tuple: (success_bool, pixel_array)
    """
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        pixel_array = dicom_data.pixel_array
        
        # Normalize to 8-bit if needed
        if pixel_array.dtype != np.uint8:
            pixel_min = pixel_array.min()
            pixel_max = pixel_array.max()
            if pixel_max > pixel_min:
                pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
        
        # Apply color mode
        if color_mode == 'gray' or (len(pixel_array.shape) == 2):
            # Convert to grayscale or keep as grayscale
            if len(pixel_array.shape) == 3:
                pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2GRAY)
        elif color_mode == 'bgr' and len(pixel_array.shape) == 3:
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2BGR)
        elif color_mode == 'rgb':
            # Keep as RGB (might need BGR conversion for display)
            pass
        elif color_mode == 'no_convert':
            # Use pixel data as-is
            pass
        elif color_mode == 'auto':
            # Auto-detect best conversion
            if hasattr(dicom_data, 'PhotometricInterpretation'):
                if dicom_data.PhotometricInterpretation == 'MONOCHROME1':
                    pixel_array = 255 - pixel_array
                elif dicom_data.PhotometricInterpretation == 'RGB':
                    # For ultrasound, often no conversion needed
                    pass
        
        # Create output filename if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(dicom_path))[0]
            output_path = f"{base_name}_{color_mode}.png"
        
        # Save the image
        success = cv2.imwrite(output_path, pixel_array)
        
        if success:
            print(f"Successfully saved: {output_path}")
            return True, pixel_array
        else:
            print(f"Failed to save: {output_path}")
            return False, pixel_array
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False, None

def callOcrModel(image):
    extractedText=extractTextFromImage(image)
    return extractedText

def main():
    """
    Main function to handle command line arguments
    """
    global img
    
    if len(sys.argv) < 2:
        print("Usage: python dicom_to_png.py <dicom_file_path> [output_png_path]")
        print("Example: python dicom_to_png.py image.dcm output.png")
        sys.exit(1)
    
    dicom_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if input file exists
    if not os.path.exists(dicom_path):
        print(f"Error: DICOM file '{dicom_path}' not found")
        sys.exit(1)
    
    # Convert DICOM to PNG
    success, converted_image = dicom_to_png(dicom_path, output_path)
    
    if not success or converted_image is None:
        print("Failed to convert DICOM file")
        sys.exit(1)
    
    # Use the converted image directly instead of reloading from file
    img = converted_image.copy()
    img=img[66:852,0:1136]
    measurementValues=callOcrModel(img)
    print("succesfull")
    # Display the image
    cv2.imshow('DICOM Image', img)

    
    print("Image displayed. Move mouse over image to see pixel values.")
    print("Press any key to exit.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()