import pydicom
import cv2
import numpy as np
import os
from paddleInference import extractTextFromImage
from dicomRegionLocation import USRegionLocation
from tableEnhancement import main
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
        print("DICOM file loaded successfully", dicom_data)
        regions_sequence = dicom_data[(0x0018, 0x6011)]  # Common tag for ultrasound regions

        # Extract pixel array
        # Handle different photometric interpretations
        
        if hasattr(dicom_data, 'PhotometricInterpretation'):
            if dicom_data.PhotometricInterpretation == 'MONOCHROME1':
                print("monochrome")
                dicom_data.PhotometricInterpretation = "YBR_FULL"
                pixel_array = dicom_data.pixel_array
                # Invert the image for MONOCHROME1 (lower values = brighter)
                # pixel_array = 255 - pixel_array
            elif dicom_data.PhotometricInterpretation == 'RGB':
                print("checkingRgb")
                # Note: OpenCV uses BGR, so we convert RGB to BGR for proper display
                dicom_data.PhotometricInterpretation = "YBR_FULL"
                bgr =dicom_data.pixel_array
                pixel_array=cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR ) 
  
        return True,pixel_array
            
    except FileNotFoundError:
        print(f"Error: DICOM file '{dicom_path}' not found")
        return False, None
    except Exception as e:
        print(f"Error converting DICOM file: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False, None

def usImageArea(dicom_path,pixel_array):
    us_region = USRegionLocation(dicom_path)
    minx0, miny0, maxx1, maxy1 = us_region.get_coordinates()
    if len(minx0) >=2:

        minX=min(minx0[0],minx0[1])    
        minY=min(miny0[0],miny0[1])
        maxX=max(maxx1[0],maxx1[1])
        maxY=max(maxy1[0],maxy1[1])
    else:
        minX=minx0[0]   
        minY=miny0[0]
        maxX=maxx1[0]
        maxY=maxy1[0]
    cropedRegion=pixel_array[minY:maxY,minX:maxX]
    return cropedRegion

def call_ocr_model(image):
    """
    Extract text from image using OCR model
    
    Args:
        image: Input image array
    
    Returns:
        Extracted text
    """
    extracted_text = extractTextFromImage(image)
    return extracted_text
def tableEnhancement(image):
    main(image)
def display_image(image, window_name='DICOM Image'):
    """
    Display image using OpenCV
    
    Args:
        image: Image array to display
        window_name: Name of the display window
    """
    cv2.imshow(window_name, image)
    print("Image displayed. Move mouse over image to see pixel values.")
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()