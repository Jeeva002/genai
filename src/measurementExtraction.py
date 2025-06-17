import sys
import os
from dicomHandler import dicom_to_png, usImageArea, call_ocr_model, display_image,tableEnhancement
from tableEnhancementOwn import enhanceTable
from textDensityPlusGeometricAnalysis import detect_tables_in_ultrasound
def main():
    """
    Main function to handle command line arguments and orchestrate the DICOM processing
    """
  
    
    dicom_path = 'c:\\Users\\Welcome\\Desktop\\IMG2DICOM\\I0000110'
    output_path = 'c:\\Users\\Welcome\\Desktop\\IMG2DICOM\\eightJune'
    

    
    # Convert DICOM to PNG
    success,converted_image = dicom_to_png(dicom_path, output_path)
    
    if not success or converted_image is None:
        print("Failed to convert DICOM file")
        sys.exit(1)
    
    # Crop the image
    cropped_image = usImageArea(dicom_path,converted_image)
    display_image(cropped_image)
    detect_tables_in_ultrasound(cropped_image)
    #enhanceTable(cropped_image)
    #tableEnhancement(cropped_image)
    # Extract text using OCR
    print("prppr")
    measurement_values = call_ocr_model(cropped_image)
    print("OCR processing successful")
    
    # Display the image
    display_image(cropped_image)

if __name__ == "__main__":
    main()